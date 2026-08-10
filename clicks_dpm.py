#!/usr/bin/env python3
"""
clicks_dpm.py

Compute detection-positive minutes (DPM) per time bucket (daily by default,
hourly optional) directly from clicks.bin (the raw combined click
detections from bcl_to_bin.py), and write a CSV report.

A minute is "positive" if it contains at least one raw click detection -
no periodicity/threshold filtering (contrast with click_periodicity_dpm.py,
which reports DPM from the Rayleigh-Z periodicity grid using a Z-score
threshold instead). Output format matches click_periodicity_dpm.py's CSV
exactly, so compare_dpm.py works against this module's output too.

Note: click times are true UTC (see bcl_to_bin.py) - the output datetime
column is explicitly UTC-labeled.

clicks.bin only contains actual click timestamps (nothing for silent
periods), so every bucket across the full [first click, last click] span
is explicitly filled in, including 0 for buckets with no clicks - otherwise
gaps (e.g. full recorder-off days) would be missing from the CSV instead of
showing as 0.

Usage:
    python clicks_dpm.py [clicks.bin] [--interval {day,hour}]
        [-o clicks_dpm.csv]

Requires: numpy
"""

import argparse
import csv
from datetime import timezone, datetime as dt

import numpy as np

from bcl_to_bin import load_clicks

BUCKET_SECONDS = {"hour": 3600, "day": 86400}


def compute_presence_minutes(times, bucket_seconds):
    """Return (bucket_start, presence_minutes): bucket_start is UTC epoch
    seconds for every bucket spanning the full range of `times` (including
    buckets with zero clicks), presence_minutes is the count of that
    bucket's minutes containing at least one click."""
    minute_key = np.floor(times / 60.0).astype(np.int64)
    positive_minutes = np.unique(minute_key)

    bucket_minutes = bucket_seconds // 60
    bucket_key = positive_minutes // bucket_minutes
    ubucket, counts = np.unique(bucket_key, return_counts=True)

    full_range = np.arange(ubucket.min(), ubucket.max() + 1)
    presence = np.zeros(full_range.shape, dtype=np.int64)
    presence[np.searchsorted(full_range, ubucket)] = counts

    bucket_start = full_range * bucket_seconds
    return bucket_start, presence


def main():
    parser = argparse.ArgumentParser(
        description="Compute cetacean detection-positive minutes per time "
                     "bucket directly from raw click detections (clicks.bin)."
    )
    parser.add_argument("input", nargs="?", default="clicks.bin",
                         help="Combined clicks binary file (default: "
                              "clicks.bin)")
    parser.add_argument("--interval", choices=("day", "hour"), default="day",
                         help="Time bucket for each output row (default: "
                              "day)")
    parser.add_argument("-o", "--output", default="clicks_dpm.csv",
                         help="Output CSV path (default: clicks_dpm.csv)")
    args = parser.parse_args()

    arr = load_clicks(args.input)
    if arr.size == 0:
        raise SystemExit(f"No records found in {args.input}")
    times = arr["time"]
    print(f"Loaded {times.size} clicks spanning "
          f"{(times.max() - times.min()) / 86400.0:.2f} days from "
          f"{args.input}")

    bucket_seconds = BUCKET_SECONDS[args.interval]
    bucket_start, presence_minutes = compute_presence_minutes(
        times, bucket_seconds)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime_utc", "cetacean_present_minutes"])
        for t, minutes in zip(bucket_start, presence_minutes):
            timestamp = dt.fromtimestamp(int(t), timezone.utc)
            writer.writerow([timestamp.isoformat(sep=" "), int(minutes)])

    interval_label = "daily" if args.interval == "day" else "hourly"
    print(f"Wrote {len(bucket_start)} {interval_label} rows to "
          f"{args.output}; {int(presence_minutes.sum())} total positive "
          f"minutes")


if __name__ == "__main__":
    main()
