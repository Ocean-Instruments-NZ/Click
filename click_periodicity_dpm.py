#!/usr/bin/env python3
"""
click_periodicity_dpm.py

Compute detection-positive minutes (DPM) per time bucket (daily by default,
hourly optional) from a click_periodicity.data file produced by
click_periodicity_process.py, and write a CSV report.

A time bin is "positive" if any candidate period's Rayleigh Z score exceeds
--threshold. A minute is positive if it contains at least one positive bin.
For each UTC bucket (day or hour), cetacean_present_minutes counts how many
of that bucket's minutes were positive (0-1440 for daily, 0-60 for hourly).

Note: click times are true UTC (see click_periodicity_process.py) - the
output datetime column is explicitly UTC-labeled.

Usage:
    python click_periodicity_dpm.py [click_periodicity.data]
        [--threshold 6.0] [--interval {day,hour}] [-o click_periodicity_dpm.csv]

Requires: numpy
"""

import argparse
import csv
from datetime import timezone, datetime as dt

import numpy as np

BUCKET_SECONDS = {"hour": 3600, "day": 86400}


def load_periodicity_data(path):
    """Load a click_periodicity.data file, upcasting z to float64 and
    reconstructing the deterministic interval_start time axis."""
    with np.load(path) as d:
        Z = d["z"].astype(np.float64)
        t0 = float(d["t0"])
        interval_sec = float(d["interval_sec"])

    interval_start = t0 + np.arange(Z.shape[0]) * interval_sec
    return Z, interval_start


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


def main():
    parser = argparse.ArgumentParser(
        description="Compute cetacean detection-positive minutes per time "
                     "bucket from a click_periodicity.data file."
    )
    parser.add_argument("input", nargs="?", default="click_periodicity.data",
                         help="Periodicity data file (default: "
                              "click_periodicity.data)")
    parser.add_argument("--threshold", type=float, default=6.0,
                         help="Rayleigh Z threshold for a positive "
                              "detection (default: 6.0)")
    parser.add_argument("--interval", choices=("day", "hour"), default="day",
                         help="Time bucket for each output row (default: "
                              "day)")
    parser.add_argument("-o", "--output", default="click_periodicity_dpm.csv",
                         help="Output CSV path (default: "
                              "click_periodicity_dpm.csv)")
    args = parser.parse_args()

    Z, interval_start = load_periodicity_data(args.input)
    print(f"Loaded {Z.shape[0]} intervals x {Z.shape[1]} periods from "
          f"{args.input}")

    bucket_seconds = BUCKET_SECONDS[args.interval]
    bucket_start, presence_minutes = compute_presence_minutes(
        interval_start, Z, args.threshold, bucket_seconds)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime_utc", "cetacean_present_minutes"])
        for t, minutes in zip(bucket_start, presence_minutes):
            timestamp = dt.fromtimestamp(int(t), timezone.utc)
            writer.writerow([timestamp.isoformat(sep=" "), int(minutes)])

    interval_label = "daily" if args.interval == "day" else "hourly"
    print(f"Wrote {len(bucket_start)} {interval_label} rows to "
          f"{args.output} (threshold={args.threshold}); "
          f"{int(presence_minutes.sum())} total positive minutes")


if __name__ == "__main__":
    main()
