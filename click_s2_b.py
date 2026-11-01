#!/usr/bin/env python3
# Copyright (c) 2026, Ocean Instruments NZ
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
click_s2_b.py

Use output of click_s1.py to calculate detection-positive-minutes (DPM) 

Writes a CSV report, click_s2_b_out.csv by default.

Rayleigh test: for a candidate inter-click interval (ICI) T, fold each click
time t to a phase
    phase = 2*pi * (t mod T) / T
and compute Z = (C**2 + S**2) / n, where C = sum(cos(phase)), S =
sum(sin(phase)), n = number of clicks. 

Large Z indicates clicks cluster at a
consistent phase within that window, i.e. periodic clicking at an ICI near T.

Analysis windows are --interval-sec wide but overlap by 50%, stepping every
--interval-sec/2.

A time bin is "positive" if any candidate ICI's Rayleigh Z score exceeds
--confidence_threshold. A minute is positive if it contains at least one
positive bin. 

For each UTC hour, cetacean_present_minutes counts how many of
that hour's minutes were positive (0-60). 

With --output_interval minute, each output row is a single minute instead, 
and cetacean_present_minutes is 1 (positive) or 0 (negative); the other columns 
are then computed over that minute alone. 

--omit-empty drops rows with zero cetacean_present_minutes.

Three more columns report periodicity/spectral character within each
bucket:
    train_count - count of those exceeding (time bin, candidate ICI) cells
    ici         - mean candidate ICI (ms) 
    frequency   - mean mid frequency (kHz) 

Note: click times are true UTC (see click_s1.py) 

--freq-min-khz/--freq-max-khz pre-filter the input clicks 

Usage:
    python click_s2_b.py [click_s1.npz] [--interval-sec 2]
        [--ici-min-ms 40] [--ici-max-ms 60] [--ici-step-ms 2]
        [--freq-min-khz N] [--freq-max-khz N]
        [--output_interval {minute,hour}] [--confidence_threshold 6.0]
        [--omit-empty] [-o click_s2_b_out.csv]

Requires: numpy
"""

import argparse
import csv
from datetime import timezone, datetime as dt

import numpy as np

from click_s1 import load_clicks, datetime64_to_epoch_seconds

BUCKET_SECONDS = {"minute": 60, "hour": 3600}


def compute_bins(times, interval_sec):
    """Assign clicks to 50%-overlapping windows (width interval_sec, step
    interval_sec/2). Each click appears once per window it falls in (up to
    two), so bincount aggregation over bin_idx works per window.

    Returns (click_idx, bin_idx, n_intervals, t0, step_sec); click_idx
    indexes the original times/mf arrays and may repeat a click.
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


def compute_periodicity(times, bin_idx, n_intervals, t0, confidence_threshold,
                         ici_min_ms, ici_max_ms, ici_step_ms):
    """Rayleigh test per (window, candidate ICI), thresholded as each ICI
    is computed so only a boolean matrix is kept (the full Z grid is never
    held in memory).

    `times` must be indexed like `bin_idx` (i.e. `times[click_idx]`).
    Returns (exceeds, periods_ms): exceeds is bool (n_intervals,
    n_periods); periods_ms is the 1D array of candidate ICIs (ms).
    """
    t_rel = times - t0

    counts = np.bincount(bin_idx, minlength=n_intervals).astype(np.float64)
    # Floor of 1 gives Z = 0 (not a divide-by-zero) for empty bins.
    counts_safe = np.maximum(counts, 1.0)

    periods_ms = np.arange(ici_min_ms,
                            ici_max_ms + ici_step_ms / 2,
                            ici_step_ms)
    periods_sec = periods_ms / 1000.0

    exceeds = np.empty((n_intervals, periods_ms.size), dtype=bool)
    for j, period in enumerate(periods_sec):
        click_phase = 2.0 * np.pi * np.mod(t_rel, period) / period
        C = np.bincount(bin_idx, weights=np.cos(click_phase),
                         minlength=n_intervals)
        S = np.bincount(bin_idx, weights=np.sin(click_phase),
                         minlength=n_intervals)
        Z_j = (C ** 2 + S ** 2) / counts_safe
        exceeds[:, j] = Z_j > confidence_threshold

    return exceeds, periods_ms


def compute_mean_mf(mf, click_idx, bin_idx, n_intervals):
    """Mean mf (Hz) per window, ignoring NaN clicks; NaN only if a window
    has no valid mf at all."""
    mf_expanded = mf[click_idx]
    valid = ~np.isnan(mf_expanded)
    sums = np.bincount(bin_idx, weights=np.where(valid, mf_expanded, 0.0),
                        minlength=n_intervals)
    counts = np.bincount(bin_idx, weights=valid.astype(np.float64),
                          minlength=n_intervals)
    with np.errstate(invalid="ignore"):
        return sums / np.where(counts == 0, np.nan, counts)


def filter_by_frequency(times, mf, freq_min_khz, freq_max_khz):
    """Keep clicks with mf within [freq_min_khz, freq_max_khz] kHz (either
    bound optional, inclusive); NaN-mf clicks are always dropped."""
    mf_khz = mf / 1000.0
    mask = ~np.isnan(mf_khz)
    if freq_min_khz is not None:
        mask &= mf_khz >= freq_min_khz
    if freq_max_khz is not None:
        mask &= mf_khz <= freq_max_khz
    return times[mask], mf[mask]


def compute_bucket_minutes(interval_start, bin_mask, bucket_seconds):
    """Return (bucket_start, minutes): per bucket, the number of minutes
    containing at least one True bin in bin_mask."""
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
    """Return (bucket_start, train_count, ici_mean, frequency_mean): per
    bucket, the count of exceeding (bin, ICI) cells and the mean ICI (ms)
    and mean mf (Hz) over them. NaN mf drops a bin from the frequency
    mean only, not the ici mean."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute a Rayleigh-test periodicity grid and mean mid "
                     "frequency from combined click detections, then "
                     "immediately report detection-positive minutes (per "
                     "hour, or per minute as 1/0), mean ICI, and mean "
                     "frequency - all in one pass, with no intermediate "
                     "click_s2.npz file."
    )
    parser.add_argument("input", nargs="?", default="click_s1.npz",
                         help="Combined clicks file from click_s1.py "
                              "(default: click_s1.npz)")
    parser.add_argument("--interval-sec", type=float, default=2.0,
                         help="Analysis window width in seconds; windows "
                              "overlap 50%%, stepping every "
                              "interval-sec/2 (default: 2.0)")
    parser.add_argument("--ici-min-ms", type=float, default=40.0,
                         help="Minimum candidate ICI (inter-click "
                              "interval) in ms (default: 40)")
    parser.add_argument("--ici-max-ms", type=float, default=60.0,
                         help="Maximum candidate ICI in ms (default: 60)")
    parser.add_argument("--ici-step-ms", type=float, default=2.0,
                         help="Candidate ICI step in ms (default: 2)")
    parser.add_argument("--freq-min-khz", type=float, default=None,
                         help="Only process clicks with mf >= this many "
                              "kHz; omit for no lower bound (default: no "
                              "filter)")
    parser.add_argument("--freq-max-khz", type=float, default=None,
                         help="Only process clicks with mf <= this many "
                              "kHz; omit for no upper bound (default: no "
                              "filter)")
    parser.add_argument("--confidence_threshold", type=float, default=6.0,
                         help="Rayleigh Z score a window must exceed for a "
                              "positive detection (default: 6.0)")
    parser.add_argument("--output_interval", choices=("minute", "hour"),
                         default="hour",
                         help="Time interval covered by each output row; in "
                              "minute mode cetacean_present_minutes is 1 "
                              "(positive) or 0 (default: hour)")
    parser.add_argument("--omit-empty", action="store_true",
                         help="Omit output rows with zero "
                              "cetacean_present_minutes (in minute mode, "
                              "keeps only positive minutes) (default: "
                              "include them)")
    parser.add_argument("-o", "--output", default="click_s2_b_out.csv",
                         help="Output CSV path (default: "
                              "click_s2_b_out.csv)")
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

    print(f"Processing {times.size} clicks spanning "
          f"{(times.max() - times.min()) / 86400.0:.2f} days")

    click_idx, bin_idx, n_intervals, t0, step_sec = compute_bins(
        times, args.interval_sec)
    print(f"Using {args.interval_sec:.1f}s analysis windows stepped every "
          f"{step_sec:.1f}s (50% overlap)")
    times_expanded = times[click_idx]

    exceeds, periods_ms = compute_periodicity(
        times_expanded, bin_idx, n_intervals, t0, args.confidence_threshold,
        args.ici_min_ms, args.ici_max_ms, args.ici_step_ms)
    print(f"Computed Z scores: {exceeds.shape[0]} windows x "
          f"{exceeds.shape[1]} candidate ICIs")

    mean_mf = compute_mean_mf(mfs, click_idx, bin_idx, n_intervals)
    interval_start = t0 + np.arange(n_intervals) * step_sec

    bucket_seconds = BUCKET_SECONDS[args.output_interval]
    bucket_start, presence_minutes = compute_bucket_minutes(
        interval_start, np.any(exceeds, axis=1), bucket_seconds)
    (ici_bucket_start, train_count, ici_mean,
     frequency_mean) = compute_mean_ici_frequency(
        interval_start, exceeds, periods_ms, mean_mf, bucket_seconds)
    assert np.array_equal(bucket_start, ici_bucket_start), (
        "bucket sets from compute_bucket_minutes and "
        "compute_mean_ici_frequency disagree")

    rows_written = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime_utc", "cetacean_present_minutes",
                          "train_count", "ici(ms)", "frequency(kHz)"])
        for (t, minutes, count, ici, freq) in zip(
                bucket_start, presence_minutes, train_count, ici_mean,
                frequency_mean):
            if args.omit_empty and minutes == 0:
                continue
            timestamp = dt.fromtimestamp(int(t), timezone.utc)
            ici_str = "" if np.isnan(ici) else f"{ici:.0f}"
            freq_str = "" if np.isnan(freq) else f"{freq / 1000.0:.0f}"
            writer.writerow([timestamp.isoformat(sep=" "), int(minutes),
                              int(count), ici_str, freq_str])
            rows_written += 1

    interval_label = ("per-minute" if args.output_interval == "minute"
                      else "hourly")
    print(f"Wrote {rows_written} {interval_label} rows to {args.output} "
          f"(confidence_threshold={args.confidence_threshold}); "
          f"{int(presence_minutes.sum())} total positive minutes")


if __name__ == "__main__":
    main()
