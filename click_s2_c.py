#!/usr/bin/env python3
"""
click_s2_c.py

Use output of click_s1.py to produce a per-bin periodicity time series.

Writes one npz file, click_s2_c_out.npz by default, with one record per
analysis time bin whose highest Rayleigh Z score (Zh) exceeds --threshold:
    t   - datetime64[us], the bin's midpoint time (true UTC)
    Zh  - the bin's highest Rayleigh Z score across all candidate periods
    mf  - the bin's mean mid frequency, in kHz (NaN if none of its clicks
          have a valid mf)
    ici - the candidate period (ms) that produced Zh - the bin's "winning"
          inter-click interval

Unlike click_s2_b.py (which aggregates into hourly detection-positive-
minutes with NBHF/NBMF/Other call-type classification), this module does
no bucketing or classification - it's a raw per-bin record of every
exceeding-threshold detection, for callers that want the underlying time
series directly.

Rayleigh test: for a candidate period T, fold each click time t to a phase
    phase = 2*pi * (t mod T) / T
and compute Z = (C**2 + S**2) / n, where C = sum(cos(phase)), S =
sum(sin(phase)), n = number of clicks. Large Z indicates clicks cluster at a
consistent phase within that interval, i.e. periodic clicking near period T.

Analysis windows are --interval-sec wide but overlap by 50%, stepping every
--interval-sec/2 - so most clicks fall in two consecutive, overlapping
windows (each window's Rayleigh test still only sees its own window's
clicks; a shared click is simply counted independently by each window it
falls in).

Note: click times are true UTC (see click_s1.py).

--freq-min-khz/--freq-max-khz pre-filter the input clicks (before any
binning) to only those whose mf (mid/peak frequency, from click_s1.py)
falls within the given range, in kHz - either bound alone is enough to
activate filtering. A click with NaN mf (missing/mismatched .dwv
snippet, or an unsaved detection) is excluded whenever a filter is
active, since its frequency can't be confirmed to be in range - this is
a real behavior change for those clicks versus running with no filter,
where they still contribute to periodicity despite their unknown
frequency.

Usage:
    python click_s2_c.py [click_s1.npz] [--interval-sec 2]
        [--period-min-ms 40] [--period-max-ms 60] [--period-step-ms 2]
        [--max-days N] [--freq-min-khz N] [--freq-max-khz N]
        [--threshold 6.0] [-o click_s2_c_out.npz]

Requires: numpy
"""

import argparse

import numpy as np

from click_s1 import load_clicks, datetime64_to_epoch_seconds, \
    epoch_seconds_to_datetime64


def compute_bins(times, interval_sec):
    """Shared time-binning logic for the periodicity heatmap and mean mf,
    so both land on identical, overlapping windows.

    Analysis windows are interval_sec wide, stepping every interval_sec/2
    (50% overlap) - since window width is always exactly 2x the step, a
    click at relative time t_rel falls in window floor(t_rel/step) and,
    except at the very start of the data, also in the window before that.
    Rather than teach every aggregation function about "up to 2 windows
    per click", each click is duplicated into one (bin_idx, click) pair
    per window it belongs to - so the existing bincount-based aggregation
    in compute_max_z/compute_mean_mf needs no change: summing over the
    expanded index naturally gives each window its own correct,
    independent set of contributing clicks, some shared with a neighbor.

    Returns (click_idx, bin_idx, n_intervals, t0, step_sec). `click_idx`
    indexes directly into the original (unsorted) times/mf arrays - e.g.
    `times[click_idx]` aligns with `bin_idx` for bincount-based
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


def compute_max_z(times, bin_idx, n_intervals, t0,
                   period_min_ms, period_max_ms, period_step_ms):
    """Compute each time bin's highest Rayleigh Z score (Zh) across all
    candidate periods, and the candidate period (ici, ms) that produced
    it, using bins already computed by `compute_bins`.

    Nothing downstream of this module needs any period's Z value once Zh
    is known, so only a running per-bin max (and its winning period
    index) is kept across the period loop - the full (n_intervals x
    n_periods) float64 Z matrix is never materialized at once, only one
    period's column at a time (freed each loop iteration).

    `times` must already be indexed the same way as `bin_idx` (as
    `compute_bins` returns via `click_idx`, e.g. `times[click_idx]`).
    Returns (Zh, ici_ms, periods_ms): Zh and ici_ms have shape
    (n_intervals,); periods_ms is 1D (ms). ici_ms is meaningless for bins
    whose Zh never exceeds a threshold, but such bins are expected to be
    dropped by the caller before use.
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

    Zh = np.full(n_intervals, -np.inf)
    winning_period_idx = np.zeros(n_intervals, dtype=np.int64)
    for j, period in enumerate(periods_sec):
        click_phase = 2.0 * np.pi * np.mod(t_rel, period) / period
        C = np.bincount(bin_idx, weights=np.cos(click_phase),
                         minlength=n_intervals)
        S = np.bincount(bin_idx, weights=np.sin(click_phase),
                         minlength=n_intervals)
        Z_j = (C ** 2 + S ** 2) / counts_safe
        better = Z_j > Zh
        Zh = np.where(better, Z_j, Zh)
        winning_period_idx = np.where(better, j, winning_period_idx)

    ici_ms = periods_ms[winning_period_idx]
    return Zh, ici_ms, periods_ms


def compute_mean_mf(mf, click_idx, bin_idx, n_intervals):
    """Mean mid frequency (`mf` field) per time bin, aligned to the same
    bins used for the periodicity heatmap. `mf` is often NaN per-click
    (missing/mismatched .dwv snippet, or an unsaved detection) - those are
    excluded from the average individually (not simply propagated as
    NaN), so a bin only comes out NaN if it has no detections with a
    valid mf at all."""
    mf_expanded = mf[click_idx]
    valid = ~np.isnan(mf_expanded)
    sums = np.bincount(bin_idx, weights=np.where(valid, mf_expanded, 0.0),
                        minlength=n_intervals)
    counts = np.bincount(bin_idx, weights=valid.astype(np.float64),
                          minlength=n_intervals)
    with np.errstate(invalid="ignore"):
        return sums / np.where(counts == 0, np.nan, counts)


def filter_by_frequency(times, mf, freq_min_khz, freq_max_khz):
    """Keep only clicks whose mf (mid/peak frequency, Hz) falls within
    [freq_min_khz, freq_max_khz] kHz (either bound optional, inclusive).
    Clicks with NaN mf are excluded whenever a filter is active, since
    their frequency can't be confirmed to be in range."""
    mf_khz = mf / 1000.0
    mask = ~np.isnan(mf_khz)
    if freq_min_khz is not None:
        mask &= mf_khz >= freq_min_khz
    if freq_max_khz is not None:
        mask &= mf_khz <= freq_max_khz
    return times[mask], mf[mask]


def main():
    parser = argparse.ArgumentParser(
        description="Compute a Rayleigh-test periodicity grid and mean mid "
                     "frequency from combined click detections, then write "
                     "a per-bin time series (t/Zh/mf/ici) of every bin "
                     "exceeding --threshold, as a single npz file."
    )
    parser.add_argument("input", nargs="?", default="click_s1.npz",
                         help="Combined clicks file from click_s1.py "
                              "(default: click_s1.npz)")
    parser.add_argument("--interval-sec", type=float, default=2.0,
                         help="Analysis window width in seconds; windows "
                              "overlap 50%%, stepping every "
                              "interval-sec/2 (default: 2.0)")
    parser.add_argument("--period-min-ms", type=float, default=40.0,
                         help="Minimum candidate period in ms (default: 40)")
    parser.add_argument("--period-max-ms", type=float, default=60.0,
                         help="Maximum candidate period in ms (default: 60)")
    parser.add_argument("--period-step-ms", type=float, default=2.0,
                         help="Candidate period step in ms (default: 2)")
    parser.add_argument("--max-days", type=float, default=None,
                         help="Limit input to the first N days of click "
                              "data (default: unlimited)")
    parser.add_argument("--freq-min-khz", type=float, default=None,
                         help="Only process clicks with mf >= this many "
                              "kHz; omit for no lower bound (default: no "
                              "filter)")
    parser.add_argument("--freq-max-khz", type=float, default=None,
                         help="Only process clicks with mf <= this many "
                              "kHz; omit for no upper bound (default: no "
                              "filter)")
    parser.add_argument("--threshold", type=float, default=6.0,
                         help="Rayleigh Z threshold for a positive "
                              "detection (default: 6.0)")
    parser.add_argument("-o", "--output", default="click_s2_c_out.npz",
                         help="Output npz path (default: "
                              "click_s2_c_out.npz)")
    args = parser.parse_args()

    arr = load_clicks(args.input)
    if arr.size == 0:
        raise SystemExit(f"No records found in {args.input}")
    times = datetime64_to_epoch_seconds(arr["t"])
    mfs = arr["mf"]

    if args.freq_min_khz is not None or args.freq_max_khz is not None:
        before = times.size
        times, mfs = filter_by_frequency(times, mfs, args.freq_min_khz,
                                          args.freq_max_khz)
        print(f"Frequency filter [{args.freq_min_khz}, {args.freq_max_khz}] "
              f"kHz: kept {times.size}/{before} clicks")
        if times.size == 0:
            raise SystemExit(
                "No records remain after applying the frequency filter")

    if args.max_days is not None:
        cutoff = times.min() + args.max_days * 86400.0
        mask = times <= cutoff
        times, mfs = times[mask], mfs[mask]
        if times.size == 0:
            raise SystemExit("No records remain after applying --max-days")

    print(f"Processing {times.size} clicks spanning "
          f"{(times.max() - times.min()) / 86400.0:.2f} days")

    click_idx, bin_idx, n_intervals, t0, step_sec = compute_bins(
        times, args.interval_sec)
    print(f"Using {args.interval_sec:.1f}s analysis windows stepped every "
          f"{step_sec:.1f}s (50% overlap)")
    times_expanded = times[click_idx]

    Zh, ici_ms, periods_ms = compute_max_z(
        times_expanded, bin_idx, n_intervals, t0,
        args.period_min_ms, args.period_max_ms, args.period_step_ms)
    print(f"Computed Z scores: {n_intervals} intervals x "
          f"{periods_ms.size} periods")

    mean_mf = compute_mean_mf(mfs, click_idx, bin_idx, n_intervals)
    interval_start = t0 + np.arange(n_intervals) * step_sec
    interval_mid = interval_start + step_sec  # window midpoint (width = 2 * step_sec)

    mask = Zh > args.threshold
    t = epoch_seconds_to_datetime64(interval_mid[mask])
    Zh_out = Zh[mask].astype(np.float32)
    mf_out = (mean_mf[mask] / 1000.0).astype(np.float32)
    ici_out = ici_ms[mask].astype(np.float32)

    # Write via an open file handle (not a bare path) so numpy doesn't
    # append its default .npz suffix if `output` doesn't already have one.
    with open(args.output, "wb") as f:
        np.savez(f, t=t, Zh=Zh_out, mf=mf_out, ici=ici_out)

    print(f"Wrote {int(mask.sum())} of {n_intervals} bins exceeding "
          f"threshold={args.threshold} to {args.output}")


if __name__ == "__main__":
    main()
