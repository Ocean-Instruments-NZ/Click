#!/usr/bin/env python3
"""
click_s2.py

Process combined cetacean click detections (produced by click_s1.py) into
a periodicity heatmap grid plus a per-bin noise floor and mean mid
frequency, using a Rayleigh test (circular statistics / epoch folding)
over a range of candidate periods. Saves the result to a compact
click_s2.npz file for click_s2_plot_results.py to render - this way the
(comparatively slow) processing step only has to run once, and
re-plotting with different decimation/output settings doesn't require
recomputing anything.

Rayleigh test: for a candidate period T, fold each click time t to a phase
    phase = 2*pi * (t mod T) / T
and compute Z = (C**2 + S**2) / n, where C = sum(cos(phase)), S =
sum(sin(phase)), n = number of clicks. Large Z indicates clicks cluster at a
consistent phase within that interval, i.e. periodic clicking near period T.

Analysis windows are --interval-sec wide (default 4s) but overlap by 50%,
stepping every --interval-sec/2 (2s by default) - so most clicks fall in
two consecutive, overlapping windows (each window's Rayleigh test still
only sees its own window's clicks; a shared click is simply counted
independently by each window it falls in). This keeps the same 2s output
row spacing as the old non-overlapping 2s default, while giving each
window more clicks (and so more statistical power) to test.

Note: click times (the .bcl files' rtime/mticks fields) are true UTC Unix
epoch seconds. The .bcl filenames are timestamped in local time instead
(~12h offset, NZ standard time) - see click_s2_plot_results.py's
click-to-copy lookup, which reads each file's actual first record rather
than trusting the filename.

Output file (click_s2.npz):
    z            - float16, shape (n_intervals, n_periods), Rayleigh Z score
    noise_level  - float16, shape (n_intervals,), mean noise level (`nl`)
                   per time bin, NaN where a bin has no clicks
    mf           - float32, shape (n_intervals,), mean mid frequency (`mf`)
                   per time bin, ignoring individual detections with no mf
                   (NaN); NaN if a bin has no detections with a valid mf.
                   float32, not float16 - mf is a frequency in Hz and real
                   values in this dataset go up to ~212 kHz, well past
                   float16's ~65504 max (which silently overflows to inf).
    periods_ms   - float16, shape (n_periods,), candidate periods in ms
    t0           - float64 scalar, first bin's start time (UTC, seconds)
    interval_sec - float64 scalar, the STEP between window starts (i.e.
                   the row spacing every downstream module reconstructs
                   the time axis from) - --interval-sec / 2, NOT the
                   --interval-sec CLI value itself, since window width and
                   row spacing are no longer the same thing.
    window_sec   - float64 scalar, the true analysis window width
                   (--interval-sec as given on the command line), saved
                   separately purely for self-description of the file.
(interval_start isn't stored - it's fully deterministic from t0/interval_sec/
n_intervals, and Unix-scale epoch seconds would overflow float16 anyway, so
storing it would be both redundant and lossy. Reconstruct it as
t0 + arange(n_intervals) * interval_sec.)

Usage:
    python click_s2.py [click_s1.npz] [--interval-sec 4]
        [--period-min-ms 40] [--period-max-ms 60] [--period-step-ms 2]
        [--max-days N] [-o click_s2.npz]

Requires: numpy
"""

import argparse
import os

import numpy as np

from click_s1 import load_clicks


def compute_bins(times, interval_sec):
    """Shared time-binning logic for the periodicity heatmap, noise level,
    and mean mf, so all three land on identical, overlapping windows.

    Analysis windows are interval_sec wide, stepping every interval_sec/2
    (50% overlap) - since window width is always exactly 2x the step, a
    click at relative time t_rel falls in window floor(t_rel/step) and,
    except at the very start of the data, also in the window before that.
    Rather than teach every aggregation function about "up to 2 windows
    per click", each click is duplicated into one (bin_idx, click) pair
    per window it belongs to - so the existing bincount-based aggregation
    in compute_periodicity/compute_noise_levels/compute_mean_mf needs no
    change: summing over the expanded index naturally gives each window
    its own correct, independent set of contributing clicks, some shared
    with a neighbor.

    Returns (click_idx, bin_idx, n_intervals, t0, step_sec). `click_idx`
    indexes directly into the original (unsorted) times/levels/mf arrays -
    e.g. `times[click_idx]` aligns with `bin_idx` for bincount-based
    aggregation, and may repeat an index for a click counted in two
    windows. step_sec = interval_sec / 2 is the spacing between
    successive window-start times.
    """
    sort_order = np.argsort(times)
    times_sorted = times[sort_order]
    t0 = times_sorted[0]
    t_rel = times_sorted - t0

    step_sec = interval_sec / 2.0
    bin_hi = np.floor(t_rel / step_sec).astype(np.int64)
    bin_lo = bin_hi - 1  # the earlier of the two overlapping windows
    n_intervals = int(bin_hi[-1]) + 1

    valid_lo = bin_lo >= 0  # false only for the earliest clicks
    click_idx = np.concatenate([sort_order, sort_order[valid_lo]])
    bin_idx = np.concatenate([bin_hi, bin_lo[valid_lo]])

    return click_idx, bin_idx, n_intervals, t0, step_sec


def compute_periodicity(times, bin_idx, n_intervals, t0,
                         period_min_ms, period_max_ms, period_step_ms):
    """Compute the Rayleigh Z score for each (time interval, candidate
    period) cell, using bins already computed by `compute_bins`.

    `times` must already be indexed the same way as `bin_idx` (as
    `compute_bins` returns via `click_idx`, e.g. `times[click_idx]`).
    Returns (Z, periods_ms) where Z has shape (n_intervals, n_periods) and
    periods_ms is 1D (ms).
    """
    t_rel = times - t0

    counts = np.bincount(bin_idx, minlength=n_intervals).astype(np.float64)
    # Bins with no clicks get C = S = 0 from bincount, so dividing by a
    # floor of 1 (instead of the true 0) naturally yields Z = 0 for them,
    # with no extra masking or divide-by-zero warnings.
    counts_safe = np.maximum(counts, 1.0)

    periods_ms = np.arange(period_min_ms,
                            period_max_ms + period_step_ms / 2,
                            period_step_ms)
    periods_sec = periods_ms / 1000.0

    Z = np.empty((n_intervals, periods_ms.size), dtype=np.float64)
    for j, period in enumerate(periods_sec):
        click_phase = 2.0 * np.pi * np.mod(t_rel, period) / period
        C = np.bincount(bin_idx, weights=np.cos(click_phase),
                         minlength=n_intervals)
        S = np.bincount(bin_idx, weights=np.sin(click_phase),
                         minlength=n_intervals)
        Z[:, j] = (C ** 2 + S ** 2) / counts_safe

    return Z, periods_ms


def compute_noise_levels(levels, click_idx, bin_idx, n_intervals):
    """Mean noise level (`nl` field) per time bin, aligned to the same bins
    used for the periodicity heatmap. Empty bins are left as NaN - there's
    no meaningful "zero" noise reading, so downstream plots should show a
    real gap rather than a misleading dip."""
    levels_expanded = levels[click_idx]
    counts = np.bincount(bin_idx, minlength=n_intervals).astype(np.float64)
    noise_sum = np.bincount(bin_idx, weights=levels_expanded,
                             minlength=n_intervals)
    with np.errstate(invalid="ignore"):
        return noise_sum / np.where(counts == 0, np.nan, counts)


def compute_mean_mf(mf, click_idx, bin_idx, n_intervals):
    """Mean mid frequency (`mf` field) per time bin, aligned to the same
    bins used for the periodicity heatmap. Unlike `level`, `mf` is often
    NaN per-click (missing/mismatched .dwv snippet, or an unsaved
    detection) - those are excluded from the average individually (not
    simply propagated as NaN), so a bin only comes out NaN if it has no
    detections with a valid mf at all."""
    mf_expanded = mf[click_idx]
    valid = ~np.isnan(mf_expanded)
    sums = np.bincount(bin_idx, weights=np.where(valid, mf_expanded, 0.0),
                        minlength=n_intervals)
    counts = np.bincount(bin_idx, weights=valid.astype(np.float64),
                          minlength=n_intervals)
    with np.errstate(invalid="ignore"):
        return sums / np.where(counts == 0, np.nan, counts)


def main():
    parser = argparse.ArgumentParser(
        description="Compute a Rayleigh-test periodicity grid, noise "
                     "floor, and mean mid frequency from combined click "
                     "detections, saved for click_s2_plot_results.py to "
                     "render."
    )
    parser.add_argument("input", nargs="?", default="click_s1.npz",
                         help="Combined clicks file from click_s1.py "
                              "(default: click_s1.npz)")
    parser.add_argument("--interval-sec", type=float, default=4.0,
                         help="Analysis window width in seconds; windows "
                              "overlap 50%%, stepping every "
                              "interval-sec/2 (default: 4.0)")
    parser.add_argument("--period-min-ms", type=float, default=40.0,
                         help="Minimum candidate period in ms (default: 40)")
    parser.add_argument("--period-max-ms", type=float, default=60.0,
                         help="Maximum candidate period in ms (default: 60)")
    parser.add_argument("--period-step-ms", type=float, default=2.0,
                         help="Candidate period step in ms (default: 2)")
    parser.add_argument("--max-days", type=float, default=None,
                         help="Limit input to the first N days of click "
                              "data (default: unlimited)")
    parser.add_argument("-o", "--output", default="click_s2.npz",
                         help="Output data file (default: click_s2.npz)")
    args = parser.parse_args()

    arr = load_clicks(args.input)
    if arr.size == 0:
        raise SystemExit(f"No records found in {args.input}")
    times = arr["time"]
    levels = arr["level"]
    mfs = arr["mf"]

    if args.max_days is not None:
        cutoff = times.min() + args.max_days * 86400.0
        mask = times <= cutoff
        times, levels, mfs = times[mask], levels[mask], mfs[mask]
        if times.size == 0:
            raise SystemExit("No records remain after applying --max-days")

    print(f"Processing {times.size} clicks spanning "
          f"{(times.max() - times.min()) / 86400.0:.2f} days")

    click_idx, bin_idx, n_intervals, t0, step_sec = compute_bins(
        times, args.interval_sec)
    print(f"Using {args.interval_sec:.1f}s analysis windows stepped every "
          f"{step_sec:.1f}s (50% overlap)")
    times_expanded = times[click_idx]

    Z, periods_ms = compute_periodicity(
        times_expanded, bin_idx, n_intervals, t0,
        args.period_min_ms, args.period_max_ms, args.period_step_ms)
    print(f"Computed Z scores: {Z.shape[0]} intervals x {Z.shape[1]} "
          f"periods")

    noise = compute_noise_levels(levels, click_idx, bin_idx, n_intervals)
    mean_mf = compute_mean_mf(mfs, click_idx, bin_idx, n_intervals)

    # Write via an open file handle (not a bare path) so numpy doesn't
    # append its default .npz suffix - we want the literal filename given.
    # NOTE: the npz key "interval_sec" holds the STEP between window
    # starts (step_sec = args.interval_sec / 2), not the window width -
    # that's what every downstream module's t0 + arange(n) * interval_sec
    # time-axis reconstruction needs. The true window width is saved
    # separately as "window_sec".
    with open(args.output, "wb") as f:
        np.savez(f, z=Z.astype(np.float16),
                  noise_level=noise.astype(np.float16),
                  mf=mean_mf.astype(np.float32),
                  periods_ms=periods_ms.astype(np.float16),
                  t0=np.float64(t0),
                  interval_sec=np.float64(step_sec),
                  window_sec=np.float64(args.interval_sec))

    print(f"Saved periodicity data to {args.output} "
          f"({os.path.getsize(args.output)} bytes)")


if __name__ == "__main__":
    main()
