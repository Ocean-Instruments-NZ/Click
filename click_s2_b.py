#!/usr/bin/env python3
"""
click_s2_b.py

Combine click_s2.py's periodicity/frequency analysis and click_s3.py's
detection-positive-minutes (DPM) / call-type reporting into a single
in-memory pass over click_s1.npz - no intermediate click_s2.npz file is
written or read. Writes one CSV report, click_s2_b_out.csv by default.

Rayleigh test: for a candidate period T, fold each click time t to a phase
    phase = 2*pi * (t mod T) / T
and compute Z = (C**2 + S**2) / n, where C = sum(cos(phase)), S =
sum(sin(phase)), n = number of clicks. Large Z indicates clicks cluster at a
consistent phase within that interval, i.e. periodic clicking near period T.
Nothing in this module's output ever needs Z's actual magnitude, only
whether it crosses --threshold, so Z is thresholded immediately after each
candidate period is computed and only a boolean exceeds-matrix is kept -
unlike click_s2.py, which saves the real Z values for click_s2_plot_results.py's
heatmap and click_periodicity_ici.py's per-bin max-Z column, this module has
no such consumer, so keeping only the boolean result cuts memory use ~8x
for the largest array this module holds (1 byte/cell vs float64's 8).

Analysis windows are --interval-sec wide but overlap by 50%, stepping every
--interval-sec/2 - so most clicks fall in two consecutive, overlapping
windows (each window's Rayleigh test still only sees its own window's
clicks; a shared click is simply counted independently by each window it
falls in).

A time bin is "positive" if any candidate period's Rayleigh Z score exceeds
--threshold. A minute is positive if it contains at least one positive bin.
For each UTC hour, cetacean_present_minutes counts how many of that hour's
minutes were positive (0-60). Reporting buckets are hardcoded to hourly
(unlike click_s3.py, --interval is not exposed here).

Each time bin is also classified into a call type, using:
    Zh - the bin's highest Z score across all candidate periods
    mf - the bin's mean mid frequency (from the click_s2.py-style analysis
         above), converted to kHz
as follows:
    if   Zh < --threshold:            no detection
    elif mf > --NHF_f_threshold:      NBHF (narrow-band high frequency)
    elif mf > --NMF_f_threshold:      NBMF (narrow-band mid frequency)
    else:                             Other
A bin that clears --threshold but has NaN mf falls through to Other, since
neither kHz comparison can be True against NaN. DPM_NBHF/DPM_NBMF/DPM_Other
report per-bucket positive minutes for each category, using the same
bin -> minute (OR) -> bucket (sum) rule as cetacean_present_minutes, just
applied to each category's bins instead of the combined Z > threshold
condition. A minute can count toward more than one category if it contains
bins of different types.

Three more columns report periodicity/spectral character within each
bucket, over every (time bin, candidate period) cell whose Z score exceeds
--threshold:
    train_count - count of those exceeding (time bin, candidate period)
                  cells
    ici         - mean candidate period (ms, rounded to a whole number) of
                  those exceeding cells
    frequency   - mean mid frequency (kHz, rounded to a whole number) of
                  the time bins contributing those cells

ici and frequency are independent aggregates - a bin can contribute to ici
even if its mf is NaN (excluded from frequency only, not from ici), since
periodicity and spectral frequency are separate measurements. train_count
is 0 and ici is blank in the CSV for a bucket with no exceeding cells;
frequency is additionally blank if none of the contributing bins have a
valid mf.

Note: click times are true UTC (see click_s1.py) - the output datetime
column is explicitly UTC-labeled.

Usage:
    python click_s2_b.py [click_s1.npz] [--interval-sec 2]
        [--period-min-ms 40] [--period-max-ms 60] [--period-step-ms 2]
        [--max-days N] [--threshold 6.0] [--NMF_f_threshold 30]
        [--NHF_f_threshold 100] [--omit-zero] [-o click_s2_b_out.csv]

Requires: numpy
"""

import argparse
import csv
from datetime import timezone, datetime as dt

import numpy as np

from click_s1 import load_clicks

BUCKET_SECONDS = 3600  # hardcoded to hourly reporting buckets


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
    in compute_periodicity/compute_mean_mf needs no change: summing over
    the expanded index naturally gives each window its own correct,
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


def compute_periodicity(times, bin_idx, n_intervals, t0, threshold,
                         period_min_ms, period_max_ms, period_step_ms):
    """Compute, for each (time interval, candidate period) cell, whether
    the Rayleigh Z score exceeds threshold, using bins already computed
    by `compute_bins`.

    Nothing downstream of this module ever needs Z's actual magnitude,
    only whether it crosses --threshold (classify_bins's detection test,
    cetacean_present_minutes' any-exceeds test, and
    compute_mean_ici_frequency's per-period exceeds mask) - so
    thresholding happens here, immediately after each period's Z column
    is computed, keeping only a boolean result. This avoids ever
    materializing the full (n_intervals x n_periods) float64 Z matrix at
    once: only one period's float64 column exists at a time (freed each
    loop iteration), and the retained `exceeds` matrix is 1 byte/cell
    (bool) instead of 8 (float64), an ~8x reduction for what's actually
    kept in memory afterward.

    `times` must already be indexed the same way as `bin_idx` (as
    `compute_bins` returns via `click_idx`, e.g. `times[click_idx]`).
    Returns (exceeds, periods_ms) where exceeds is a bool array of shape
    (n_intervals, n_periods) and periods_ms is 1D (ms).
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

    exceeds = np.empty((n_intervals, periods_ms.size), dtype=bool)
    for j, period in enumerate(periods_sec):
        click_phase = 2.0 * np.pi * np.mod(t_rel, period) / period
        C = np.bincount(bin_idx, weights=np.cos(click_phase),
                         minlength=n_intervals)
        S = np.bincount(bin_idx, weights=np.sin(click_phase),
                         minlength=n_intervals)
        Z_j = (C ** 2 + S ** 2) / counts_safe
        exceeds[:, j] = Z_j > threshold

    return exceeds, periods_ms


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


def compute_bucket_minutes(interval_start, bin_mask, bucket_seconds):
    """Return (bucket_start, minutes): per-bucket count of minutes
    containing at least one bin where bin_mask is True."""
    minute_key = np.floor(interval_start / 60.0).astype(np.int64)
    umin, min_inv = np.unique(minute_key, return_inverse=True)
    minute_positive = np.bincount(
        min_inv, weights=bin_mask.astype(np.float64)) > 0

    bucket_key = umin // (bucket_seconds // 60)
    ubucket, bucket_inv = np.unique(bucket_key, return_inverse=True)
    minutes = np.bincount(
        bucket_inv, weights=minute_positive.astype(np.float64)).astype(int)

    return ubucket * bucket_seconds, minutes


def compute_mean_ici_frequency(interval_start, exceeds, periods_ms, mf,
                                bucket_seconds):
    """Return (bucket_start, train_count, ici_mean, frequency_mean):
    per-bucket count of exceeding (bin, period) cells, and the mean
    candidate period and mean mid frequency over those cells, where Z
    exceeds threshold (already applied - `exceeds` is a boolean mask, see
    compute_periodicity). ici and frequency are independent aggregates -
    a bin can contribute to ici even if its mf is NaN (excluded from
    frequency only), since periodicity and spectral frequency are separate
    measurements."""
    exceed_count = exceeds.sum(axis=1).astype(np.float64)
    ici_sum = exceeds.astype(np.float64) @ periods_ms

    mf_valid = ~np.isnan(mf)
    freq_count = np.where(mf_valid, exceed_count, 0.0)
    freq_sum = np.where(mf_valid, exceed_count * mf, 0.0)

    bucket_key = np.floor(interval_start / bucket_seconds).astype(np.int64)
    ubucket, inv = np.unique(bucket_key, return_inverse=True)

    total_exceed = np.bincount(inv, weights=exceed_count, minlength=len(ubucket))
    total_ici_sum = np.bincount(inv, weights=ici_sum, minlength=len(ubucket))
    total_freq_count = np.bincount(inv, weights=freq_count, minlength=len(ubucket))
    total_freq_sum = np.bincount(inv, weights=freq_sum, minlength=len(ubucket))

    with np.errstate(invalid="ignore"):
        ici_mean = np.where(total_exceed > 0, total_ici_sum / total_exceed, np.nan)
        frequency_mean = np.where(total_freq_count > 0,
                                   total_freq_sum / total_freq_count, np.nan)

    return (ubucket * bucket_seconds, total_exceed.astype(int), ici_mean,
            frequency_mean)


def classify_bins(exceeds, mf, nmf_threshold_khz, nhf_threshold_khz):
    """Classify each time bin into no-detection / NBHF / NBMF / Other,
    based on whether any candidate period's Rayleigh Z score exceeds
    threshold (already applied - `exceeds` is a boolean mask, see
    compute_periodicity) and the bin's mean mid frequency (mf, Hz,
    converted to kHz for the threshold comparisons).

    Returns (is_nbhf, is_nbmf, is_other). A bin with NaN mf that otherwise
    qualifies as a detection falls through to Other, since neither kHz
    comparison can be True against NaN."""
    mf_khz = mf / 1000.0

    is_detection = exceeds.any(axis=1)
    is_nbhf = is_detection & (mf_khz > nhf_threshold_khz)
    is_nbmf = is_detection & ~is_nbhf & (mf_khz > nmf_threshold_khz)
    is_other = is_detection & ~is_nbhf & ~is_nbmf

    return is_nbhf, is_nbmf, is_other


def main():
    parser = argparse.ArgumentParser(
        description="Compute a Rayleigh-test periodicity grid and mean mid "
                     "frequency from combined click detections, then "
                     "immediately report hourly detection-positive minutes "
                     "(overall and per call-type category), mean ICI, and "
                     "mean frequency - all in one pass, with no "
                     "intermediate click_s2.npz file."
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
    parser.add_argument("--threshold", type=float, default=6.0,
                         help="Rayleigh Z threshold for a positive "
                              "detection (default: 6.0)")
    parser.add_argument("--NMF_f_threshold", type=float, default=30.0,
                         help="Mid-frequency threshold in kHz for the "
                              "NBMF call-type category (default: 30)")
    parser.add_argument("--NHF_f_threshold", type=float, default=100.0,
                         help="High-frequency threshold in kHz for the "
                              "NBHF call-type category (default: 100)")
    parser.add_argument("--omit-zero", action="store_true",
                         help="Omit rows with zero "
                              "cetacean_present_minutes from the output "
                              "(default: include them)")
    parser.add_argument("-o", "--output", default="click_s2_b_out.csv",
                         help="Output CSV path (default: "
                              "click_s2_b_out.csv)")
    args = parser.parse_args()

    arr = load_clicks(args.input)
    if arr.size == 0:
        raise SystemExit(f"No records found in {args.input}")
    times = arr["time"]
    mfs = arr["mf"]

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

    exceeds, periods_ms = compute_periodicity(
        times_expanded, bin_idx, n_intervals, t0, args.threshold,
        args.period_min_ms, args.period_max_ms, args.period_step_ms)
    print(f"Computed Z scores: {exceeds.shape[0]} intervals x "
          f"{exceeds.shape[1]} periods")

    mean_mf = compute_mean_mf(mfs, click_idx, bin_idx, n_intervals)
    interval_start = t0 + np.arange(n_intervals) * step_sec

    bucket_start, presence_minutes = compute_bucket_minutes(
        interval_start, np.any(exceeds, axis=1), BUCKET_SECONDS)
    (ici_bucket_start, train_count, ici_mean,
     frequency_mean) = compute_mean_ici_frequency(
        interval_start, exceeds, periods_ms, mean_mf, BUCKET_SECONDS)
    assert np.array_equal(bucket_start, ici_bucket_start), (
        "bucket sets from compute_bucket_minutes and "
        "compute_mean_ici_frequency disagree")

    is_nbhf, is_nbmf, is_other = classify_bins(
        exceeds, mean_mf, args.NMF_f_threshold, args.NHF_f_threshold)
    _, nbhf_minutes = compute_bucket_minutes(
        interval_start, is_nbhf, BUCKET_SECONDS)
    _, nbmf_minutes = compute_bucket_minutes(
        interval_start, is_nbmf, BUCKET_SECONDS)
    _, other_minutes = compute_bucket_minutes(
        interval_start, is_other, BUCKET_SECONDS)

    rows_written = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime_utc", "cetacean_present_minutes",
                          "DPM_NBHF", "DPM_NBMF", "DPM_Other",
                          "train_count", "ici", "frequency"])
        for (t, minutes, nbhf, nbmf, other, count, ici,
             freq) in zip(bucket_start, presence_minutes, nbhf_minutes,
                           nbmf_minutes, other_minutes, train_count,
                           ici_mean, frequency_mean):
            if args.omit_zero and minutes == 0:
                continue
            timestamp = dt.fromtimestamp(int(t), timezone.utc)
            ici_str = "" if np.isnan(ici) else f"{ici:.0f}"
            freq_str = "" if np.isnan(freq) else f"{freq / 1000.0:.0f}"
            writer.writerow([timestamp.isoformat(sep=" "), int(minutes),
                              int(nbhf), int(nbmf), int(other), int(count),
                              ici_str, freq_str])
            rows_written += 1

    print(f"Wrote {rows_written} hourly rows to {args.output} "
          f"(threshold={args.threshold}); "
          f"{int(presence_minutes.sum())} total positive minutes")


if __name__ == "__main__":
    main()
