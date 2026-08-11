#!/usr/bin/env python3
"""
click_s3.py

Compute detection-positive minutes (DPM) per time bucket (hourly by default,
daily optional) from a click_s2.npz file produced by click_s2.py, and
write a CSV report.

A time bin is "positive" if any candidate period's Rayleigh Z score exceeds
--threshold. A minute is positive if it contains at least one positive bin.
For each UTC bucket (day or hour), cetacean_present_minutes counts how many
of that bucket's minutes were positive (0-1440 for daily, 0-60 for hourly).

Three more columns report periodicity/spectral character within each
bucket, over every (time bin, candidate period) cell whose Z score exceeds
--threshold:
    train_count - count of those exceeding (time bin, candidate period)
                  cells
    ici         - mean candidate period (ms, rounded to a whole number) of
                  those exceeding cells
    frequency   - mean mid frequency (kHz, rounded to a whole number; from
                  click_s2.py's per-bin mean mf) of the time bins
                  contributing those cells

ici and frequency are independent aggregates - a bin can contribute to ici
even if its mf is NaN (excluded from frequency only, not from ici), since
periodicity and spectral frequency are separate measurements. train_count
is 0 and ici is blank in the CSV for a bucket with no exceeding cells;
frequency is additionally blank if none of the contributing bins have a
valid mf.

Note: click times are true UTC (see click_s1.py) - the output datetime
column is explicitly UTC-labeled.

Usage:
    python click_s3.py [click_s2.npz]
        [--threshold 6.0] [--interval {day,hour}] [--omit-zero]
        [-o click_s3.csv]

Requires: numpy
"""

import argparse
import csv
from datetime import timezone, datetime as dt

import numpy as np

BUCKET_SECONDS = {"hour": 3600, "day": 86400}


def load_periodicity_data(path):
    """Load a click_s2.npz file, upcasting z/periods_ms/mf to float64 and
    reconstructing the deterministic interval_start time axis."""
    with np.load(path) as d:
        Z = d["z"].astype(np.float64)
        periods_ms = d["periods_ms"].astype(np.float64)
        mf = d["mf"].astype(np.float64)
        t0 = float(d["t0"])
        interval_sec = float(d["interval_sec"])

    interval_start = t0 + np.arange(Z.shape[0]) * interval_sec
    return Z, periods_ms, mf, interval_start


def compute_presence_minutes(interval_start, Z, threshold, bucket_seconds):
    """Return (bucket_start, presence_minutes): bucket_start is UTC epoch
    seconds for each time bucket that appears in the data, presence_minutes
    is the count of that bucket's minutes containing at least one bin
    where any period's Z score exceeds threshold."""
    bin_positive = np.any(Z > threshold, axis=1)

    minute_key = np.floor(interval_start / 60.0).astype(np.int64)
    umin, min_inv = np.unique(minute_key, return_inverse=True)
    minute_positive = np.bincount(
        min_inv, weights=bin_positive.astype(np.float64)) > 0

    bucket_key = umin // (bucket_seconds // 60)
    ubucket, bucket_inv = np.unique(bucket_key, return_inverse=True)
    presence_minutes = np.bincount(
        bucket_inv, weights=minute_positive.astype(np.float64)).astype(int)

    bucket_start = ubucket * bucket_seconds
    return bucket_start, presence_minutes


def compute_mean_ici_frequency(interval_start, Z, periods_ms, mf, threshold,
                                bucket_seconds):
    """Return (bucket_start, train_count, ici_mean, frequency_mean):
    per-bucket count of exceeding (bin, period) cells, and the mean
    candidate period and mean mid frequency over those cells, where Z
    exceeds threshold. ici and frequency are independent aggregates - a
    bin can contribute to ici even if its mf is NaN (excluded from
    frequency only), since periodicity and spectral frequency are separate
    measurements."""
    exceeds = Z > threshold
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
        description="Compute cetacean detection-positive minutes, mean ICI, "
                     "and mean frequency per time bucket from a "
                     "click_s2.npz file."
    )
    parser.add_argument("input", nargs="?", default="click_s2.npz",
                         help="Periodicity data file (default: "
                              "click_s2.npz)")
    parser.add_argument("--threshold", type=float, default=6.0,
                         help="Rayleigh Z threshold for a positive "
                              "detection (default: 6.0)")
    parser.add_argument("--interval", choices=("day", "hour"), default="hour",
                         help="Time bucket for each output row (default: "
                              "hour)")
    parser.add_argument("--omit-zero", action="store_true",
                         help="Omit rows with zero "
                              "cetacean_present_minutes from the output "
                              "(default: include them)")
    parser.add_argument("-o", "--output", default="click_s3.csv",
                         help="Output CSV path (default: click_s3.csv)")
    args = parser.parse_args()

    Z, periods_ms, mf, interval_start = load_periodicity_data(args.input)
    print(f"Loaded {Z.shape[0]} intervals x {Z.shape[1]} periods from "
          f"{args.input}")

    bucket_seconds = BUCKET_SECONDS[args.interval]
    bucket_start, presence_minutes = compute_presence_minutes(
        interval_start, Z, args.threshold, bucket_seconds)
    (ici_bucket_start, train_count, ici_mean,
     frequency_mean) = compute_mean_ici_frequency(
        interval_start, Z, periods_ms, mf, args.threshold, bucket_seconds)
    assert np.array_equal(bucket_start, ici_bucket_start), (
        "bucket sets from compute_presence_minutes and "
        "compute_mean_ici_frequency disagree")

    rows_written = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime_utc", "cetacean_present_minutes",
                          "train_count", "ici", "frequency"])
        for t, minutes, count, ici, freq in zip(
                bucket_start, presence_minutes, train_count, ici_mean,
                frequency_mean):
            if args.omit_zero and minutes == 0:
                continue
            timestamp = dt.fromtimestamp(int(t), timezone.utc)
            ici_str = "" if np.isnan(ici) else f"{ici:.0f}"
            freq_str = "" if np.isnan(freq) else f"{freq / 1000.0:.0f}"
            writer.writerow([timestamp.isoformat(sep=" "), int(minutes),
                              int(count), ici_str, freq_str])
            rows_written += 1

    interval_label = "daily" if args.interval == "day" else "hourly"
    print(f"Wrote {rows_written} {interval_label} rows to "
          f"{args.output} (threshold={args.threshold}); "
          f"{int(presence_minutes.sum())} total positive minutes")


if __name__ == "__main__":
    main()
