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
    phase        - float16, shape (n_intervals, n_periods), mean resultant-
                   vector phase (atan2(S, C)) in degrees for each Z cell -
                   where in the candidate period's cycle clicks cluster.
                   0 for empty bins (no special masking needed).
    t0           - float64 scalar, first bin's start time (UTC, seconds)
    interval_sec - float64 scalar, bin width in seconds
(interval_start isn't stored - it's fully deterministic from t0/interval_sec/
n_intervals, and Unix-scale epoch seconds would overflow float16 anyway, so
storing it would be both redundant and lossy. Reconstruct it as
t0 + arange(n_intervals) * interval_sec.)

Usage:
    python click_s2.py [click_s1.npz] [--interval-sec 2]
        [--period-min-ms 40] [--period-max-ms 60] [--period-step-ms 2]
        [--max-days N] [-o click_s2.npz]

Requires: numpy
"""

import argparse
import os

import numpy as np

from click_s1 import load_clicks


def compute_bins(times, interval_sec):
    """Shared time-binning logic for both the periodicity heatmap and the
    noise-level line, so the two always land on identical columns.

    Returns (sort_order, bin_idx, n_intervals, t0). `sort_order` is
    `times`' sort permutation - apply it to any parallel array (e.g. noise
    levels) before indexing with `bin_idx`.
    """
    sort_order = np.argsort(times)
    times_sorted = times[sort_order]
    t0 = times_sorted[0]
    t_rel = times_sorted - t0

    bin_idx = np.floor(t_rel / interval_sec).astype(np.int64)
    n_intervals = int(bin_idx[-1]) + 1
    return sort_order, bin_idx, n_intervals, t0


def compute_periodicity(times, bin_idx, n_intervals, t0, interval_sec,
                         period_min_ms, period_max_ms, period_step_ms):
    """Compute the Rayleigh Z score and mean resultant-vector phase for
    each (time interval, candidate period) cell, using bins already
    computed by `compute_bins`.

    `times` must already be sorted (as `compute_bins` returns via
    `sort_order`). Returns (Z, periods_ms, phase_deg) where Z and
    phase_deg have shape (n_intervals, n_periods) and periods_ms is 1D
    (ms). phase_deg is the resultant vector's angle atan2(S, C) in
    degrees - where in the candidate period's cycle clicks cluster,
    complementing Z's "how strongly". Empty bins get C = S = 0, and
    atan2(0, 0) is 0 with no warning, so they naturally read phase = 0.
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
    phase_deg = np.empty((n_intervals, periods_ms.size), dtype=np.float64)
    for j, period in enumerate(periods_sec):
        click_phase = 2.0 * np.pi * np.mod(t_rel, period) / period
        C = np.bincount(bin_idx, weights=np.cos(click_phase),
                         minlength=n_intervals)
        S = np.bincount(bin_idx, weights=np.sin(click_phase),
                         minlength=n_intervals)
        Z[:, j] = (C ** 2 + S ** 2) / counts_safe
        phase_deg[:, j] = np.degrees(np.arctan2(S, C))

    return Z, periods_ms, phase_deg


def compute_noise_levels(levels, sort_order, bin_idx, n_intervals):
    """Mean noise level (`nl` field) per time bin, aligned to the same bins
    used for the periodicity heatmap. Empty bins are left as NaN - there's
    no meaningful "zero" noise reading, so downstream plots should show a
    real gap rather than a misleading dip."""
    levels_sorted = levels[sort_order]
    counts = np.bincount(bin_idx, minlength=n_intervals).astype(np.float64)
    noise_sum = np.bincount(bin_idx, weights=levels_sorted,
                             minlength=n_intervals)
    with np.errstate(invalid="ignore"):
        return noise_sum / np.where(counts == 0, np.nan, counts)


def compute_mean_mf(mf, sort_order, bin_idx, n_intervals):
    """Mean mid frequency (`mf` field) per time bin, aligned to the same
    bins used for the periodicity heatmap. Unlike `level`, `mf` is often
    NaN per-click (missing/mismatched .dwv snippet, or an unsaved
    detection) - those are excluded from the average individually (not
    simply propagated as NaN), so a bin only comes out NaN if it has no
    detections with a valid mf at all."""
    mf_sorted = mf[sort_order]
    valid = ~np.isnan(mf_sorted)
    sums = np.bincount(bin_idx, weights=np.where(valid, mf_sorted, 0.0),
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
    parser.add_argument("--interval-sec", type=float, default=2.0,
                         help="Time bin width in seconds (default: 2.0)")
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

    sort_order, bin_idx, n_intervals, t0 = compute_bins(
        times, args.interval_sec)
    times_sorted = times[sort_order]

    Z, periods_ms, phase_deg = compute_periodicity(
        times_sorted, bin_idx, n_intervals, t0, args.interval_sec,
        args.period_min_ms, args.period_max_ms, args.period_step_ms)
    print(f"Computed Z scores: {Z.shape[0]} intervals x {Z.shape[1]} "
          f"periods")

    noise = compute_noise_levels(levels, sort_order, bin_idx, n_intervals)
    mean_mf = compute_mean_mf(mfs, sort_order, bin_idx, n_intervals)

    # Write via an open file handle (not a bare path) so numpy doesn't
    # append its default .npz suffix - we want the literal filename given.
    with open(args.output, "wb") as f:
        np.savez(f, z=Z.astype(np.float16),
                  noise_level=noise.astype(np.float16),
                  mf=mean_mf.astype(np.float32),
                  periods_ms=periods_ms.astype(np.float16),
                  phase=phase_deg.astype(np.float16),
                  t0=np.float64(t0),
                  interval_sec=np.float64(args.interval_sec))

    print(f"Saved periodicity data to {args.output} "
          f"({os.path.getsize(args.output)} bytes)")


if __name__ == "__main__":
    main()
