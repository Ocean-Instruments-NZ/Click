# About this project

Tools for processing SoundTrap hydrophone click-detector output (`Hydrophone/*.bcl` +
`*.dwv`) into cetacean detection reports: raw click detections are combined into one
file, then analysed for periodic click trains (echolocation) and summarised as
detection-positive minutes.

All click times are true UTC throughout this pipeline. Note: `.bcl`/`.dwv` **filenames**
are timestamped in local time instead (~12h offset, NZ standard time) — an easy source of
confusion, called out in the module docstrings.

Both modules are released under the BSD 3-Clause license (Copyright (c) 2026, Ocean
Instruments NZ); the license text is at the top of each `.py` file.

## Pipeline overview

```
Hydrophone/*.bcl, *.dwv (raw SoundTrap detector logs + audio snippets)
        |
        v
   click_s1.py  -->  click_s1.npz
                          |
                          v
                     click_s2_b.py  -->  click_s2_b_out.csv
```

Requires: Python 3 with `numpy` and `pandas` (`pandas` is only used by `click_s1.py`).

## Modules

### `click_s1.py` — the entry point

Usage: `python click_s1.py input_folder [-o output.npz] [--max-files N]`

Reads every `*.bcl` file in `input_folder` (required; sorted chronologically by the
timestamp embedded in each filename) and combines all detections into a single `.npz`
file, `click_s1.npz` by default. `--max-files` limits the run to the first N files.
Three arrays, one row per detection:
- `t` — true UTC, `datetime64[us]` (microsecond precision)
- `level` — noise level in centibels at the time of detection
- `mf` — "mid frequency" in Hz: the peak-power frequency (DC bin excluded) from an FFT of
  the detection's audio snippet in the matching `.dwv` file (one 864-sample snippet per
  saved detection). `NaN` where no snippet is available (missing `.dwv`, snippet not saved
  for that detection, or a snippet-count mismatch for that file).

All detections are accumulated in memory and written in a single `np.savez()` call at the
end (the full dataset is only a few MB).

It also provides the shared helpers used by downstream modules: `load_clicks()` (the
canonical reader for `click_s1.npz`), and `datetime64_to_epoch_seconds()` /
`epoch_seconds_to_datetime64()` for converting `t` to and from float64 Unix seconds.

### `click_s2_b.py` — periodicity analysis and detection-positive-minutes report

Usage: `python click_s2_b.py [click_s1.npz] [options] [-o click_s2_b_out.csv]`

Reads `click_s1.npz` and, in one in-memory pass (no intermediate files), looks for
periodic click trains with a Rayleigh test (circular statistics / "epoch folding"):
- Clicks are grouped into analysis windows `--interval-sec` wide (default 2 s) that
  overlap by 50%, stepping every `--interval-sec`/2, so most clicks fall in two windows.
- For each candidate inter-click interval (ICI) from `--ici-min-ms` to `--ici-max-ms` in
  steps of `--ici-step-ms` (defaults 40–60 ms, step 2), the window's clicks are folded onto
  that ICI and a Z score is computed. A window is **positive** if any candidate ICI gives
  Z above `--confidence_threshold` (default 6.0). Only a boolean exceeds-matrix is kept,
  never the full Z grid.
- A **minute** is positive if it contains at least one positive window.

`--output_interval {hour,minute}` (default `hour`) sets what each output row covers. In
`hour` mode `cetacean_present_minutes` is the number of positive minutes in that UTC hour
(0–60). In `minute` mode each row is one minute and the value is 1 (positive) or 0
(negative). `--omit-empty` drops rows where `cetacean_present_minutes` is 0.

Output CSV columns (the last three are computed over every exceeding (window, ICI) cell in
the row's interval):
- `datetime_utc` — start of the hour/minute, UTC
- `cetacean_present_minutes` — as above
- `train_count` — number of exceeding (window, ICI) cells
- `ici(ms)` — their mean ICI, rounded to a whole ms (blank if none)
- `frequency(kHz)` — mean of the contributing windows' mean `mf`, rounded to a whole kHz
  (blank if none have a valid `mf`)

`--freq-min-khz` / `--freq-max-khz` pre-filter the input clicks (before any windowing) to
those whose `mf` is in range; either bound alone activates filtering, and clicks with
`NaN` `mf` are excluded whenever a filter is active.

## Shared conventions

- **Time**: stored as UTC `datetime64[us]` in `click_s1.npz`; analysis code converts to
  float64 Unix epoch seconds for arithmetic. CSV outputs use an explicit `datetime_utc`
  column.
- **Dense time series**: every hour/minute across the data's span appears in the output,
  including ones with no detections, unless `--omit-empty` is used.
- **`load_clicks()`**: the canonical way to read `click_s1.npz`, defined in `click_s1.py`
  and reused (not duplicated) by downstream modules.

