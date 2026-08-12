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

Each time bin is also classified into a call type, using:
    Zh - the bin's highest Z score across all candidate periods
    Zp - the phase (degrees, from click_s2.py's phase array) at the
         candidate period that produced Zh
    mf - the bin's mean mid frequency (from click_s2.py), converted to kHz
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
    frequency   - mean mid frequency (kHz, rounded to a whole number; from
                  click_s2.py's per-bin mean mf) of the time bins
                  contributing those cells

ici and frequency are independent aggregates - a bin can contribute to ici
even if its mf is NaN (excluded from frequency only, not from ici), since
periodicity and spectral frequency are separate measurements. train_count
is 0 and ici is blank in the CSV for a bucket with no exceeding cells;
frequency is additionally blank if none of the contributing bins have a
valid mf.

sonar_risk flags buckets where the detection phase (Zp) drifts suspiciously
linearly with time, characteristic of an active sonar sweep rather than
genuine biological clicking. Within each bucket, only bins classified as a
detection (Zh >= --threshold) are split into non-overlapping
--sonar_interval_m-minute sub-windows (aligned to the bucket's start); each
sub-window with >=2 detection bins gets a Spearman correlation between Zp
and timestamp; sonar_risk is the maximum absolute correlation across those
sub-windows, blank if the bucket has no sub-window with enough detections.
Note this uses ordinary (non-circular) Spearman correlation on Zp, which
wraps at +/-180 degrees - a known limitation of the method as specified.

Note: click times are true UTC (see click_s1.py) - the output datetime
column is explicitly UTC-labeled.

Usage:
    python click_s3.py [click_s2.npz]
        [--threshold 6.0] [--interval {day,hour}] [--omit-zero]
        [--NMF_f_threshold 30] [--NHF_f_threshold 100]
        [--sonar_interval_m 5] [-o click_s3.csv]

Requires: numpy, scipy
"""

import argparse
import csv
import warnings
from datetime import timezone, datetime as dt

import numpy as np
from scipy import stats

BUCKET_SECONDS = {"hour": 3600, "day": 86400}


def load_periodicity_data(path):
    """Load a click_s2.npz file, upcasting z/periods_ms/mf/phase to
    float64 and reconstructing the deterministic interval_start time
    axis."""
    with np.load(path) as d:
        Z = d["z"].astype(np.float64)
        periods_ms = d["periods_ms"].astype(np.float64)
        mf = d["mf"].astype(np.float64)
        phase = d["phase"].astype(np.float64)
        t0 = float(d["t0"])
        interval_sec = float(d["interval_sec"])

    interval_start = t0 + np.arange(Z.shape[0]) * interval_sec
    return Z, periods_ms, mf, phase, interval_start


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


def classify_bins(Z, phase, mf, threshold, nmf_threshold_khz,
                   nhf_threshold_khz):
    """Classify each time bin into no-detection / NBHF / NBMF / Other,
    based on its highest Rayleigh Z score (Zh) across candidate periods,
    the phase at that winning period (Zp), and the bin's mean mid
    frequency (mf, Hz, converted to kHz for the threshold comparisons).

    Returns (Zh, Zp, is_detection, is_nbhf, is_nbmf, is_other). A bin with
    NaN mf that otherwise qualifies as a detection falls through to Other,
    since neither kHz comparison can be True against NaN."""
    n = Z.shape[0]
    winning = Z.argmax(axis=1)
    rows = np.arange(n)
    Zh = Z[rows, winning]
    Zp = phase[rows, winning]
    mf_khz = mf / 1000.0

    is_detection = Zh >= threshold
    is_nbhf = is_detection & (mf_khz > nhf_threshold_khz)
    is_nbmf = is_detection & ~is_nbhf & (mf_khz > nmf_threshold_khz)
    is_other = is_detection & ~is_nbhf & ~is_nbmf

    return Zh, Zp, is_detection, is_nbhf, is_nbmf, is_other


def compute_sonar_risk(interval_start, is_detection, Zp, bucket_seconds,
                        sonar_interval_sec):
    """Return (bucket_start, sonar_risk): per-bucket max |Spearman rho|
    between Zp and timestamp, computed within non-overlapping
    sonar_interval_sec-wide sub-windows (aligned to each bucket's start)
    and maximized across those sub-windows. Only detection bins
    (is_detection) contribute. NaN for a bucket with no sub-window
    containing at least 2 detection bins."""
    det_t = interval_start[is_detection]
    det_zp = Zp[is_detection]

    all_bucket_key = np.floor(interval_start / bucket_seconds).astype(np.int64)
    ubucket = np.unique(all_bucket_key)
    det_bucket_key = np.floor(det_t / bucket_seconds).astype(np.int64)

    sonar_risk = np.full(len(ubucket), np.nan)
    for i, b in enumerate(ubucket):
        mask = det_bucket_key == b
        if not np.any(mask):
            continue
        t_bucket, zp_bucket = det_t[mask], det_zp[mask]
        sub_idx = np.floor((t_bucket - b * bucket_seconds)
                            / sonar_interval_sec).astype(np.int64)
        best = np.nan
        for s in np.unique(sub_idx):
            sel = sub_idx == s
            if sel.sum() < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = stats.spearmanr(t_bucket[sel], zp_bucket[sel])
            if np.isnan(rho):
                continue
            best = abs(rho) if np.isnan(best) else max(best, abs(rho))
        sonar_risk[i] = best

    return ubucket * bucket_seconds, sonar_risk


def main():
    parser = argparse.ArgumentParser(
        description="Compute cetacean detection-positive minutes (overall "
                     "and per call-type category), mean ICI, mean "
                     "frequency, and sonar-drift risk per time bucket from "
                     "a click_s2.npz file."
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
    parser.add_argument("--NMF_f_threshold", type=float, default=30.0,
                         help="Mid-frequency threshold in kHz for the "
                              "NBMF call-type category (default: 30)")
    parser.add_argument("--NHF_f_threshold", type=float, default=100.0,
                         help="High-frequency threshold in kHz for the "
                              "NBHF call-type category (default: 100)")
    parser.add_argument("--sonar_interval_m", type=float, default=5.0,
                         help="Sub-window width in minutes for the "
                              "sonar_risk Spearman correlation (default: "
                              "5)")
    parser.add_argument("-o", "--output", default="click_s3.csv",
                         help="Output CSV path (default: click_s3.csv)")
    args = parser.parse_args()

    Z, periods_ms, mf, phase, interval_start = load_periodicity_data(
        args.input)
    print(f"Loaded {Z.shape[0]} intervals x {Z.shape[1]} periods from "
          f"{args.input}")

    bucket_seconds = BUCKET_SECONDS[args.interval]
    bucket_start, presence_minutes = compute_bucket_minutes(
        interval_start, np.any(Z > args.threshold, axis=1), bucket_seconds)
    (ici_bucket_start, train_count, ici_mean,
     frequency_mean) = compute_mean_ici_frequency(
        interval_start, Z, periods_ms, mf, args.threshold, bucket_seconds)
    assert np.array_equal(bucket_start, ici_bucket_start), (
        "bucket sets from compute_bucket_minutes and "
        "compute_mean_ici_frequency disagree")

    Zh, Zp, is_detection, is_nbhf, is_nbmf, is_other = classify_bins(
        Z, phase, mf, args.threshold, args.NMF_f_threshold,
        args.NHF_f_threshold)
    _, nbhf_minutes = compute_bucket_minutes(
        interval_start, is_nbhf, bucket_seconds)
    _, nbmf_minutes = compute_bucket_minutes(
        interval_start, is_nbmf, bucket_seconds)
    _, other_minutes = compute_bucket_minutes(
        interval_start, is_other, bucket_seconds)

    sonar_bucket_start, sonar_risk = compute_sonar_risk(
        interval_start, is_detection, Zp, bucket_seconds,
        args.sonar_interval_m * 60.0)
    assert np.array_equal(bucket_start, sonar_bucket_start), (
        "bucket sets from compute_bucket_minutes and compute_sonar_risk "
        "disagree")

    rows_written = 0
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime_utc", "cetacean_present_minutes",
                          "DPM_NBHF", "DPM_NBMF", "DPM_Other",
                          "train_count", "ici", "frequency", "sonar_risk"])
        for (t, minutes, nbhf, nbmf, other, count, ici, freq,
             risk) in zip(bucket_start, presence_minutes, nbhf_minutes,
                           nbmf_minutes, other_minutes, train_count,
                           ici_mean, frequency_mean, sonar_risk):
            if args.omit_zero and minutes == 0:
                continue
            timestamp = dt.fromtimestamp(int(t), timezone.utc)
            ici_str = "" if np.isnan(ici) else f"{ici:.0f}"
            freq_str = "" if np.isnan(freq) else f"{freq / 1000.0:.0f}"
            risk_str = "" if np.isnan(risk) else f"{risk:.2f}"
            writer.writerow([timestamp.isoformat(sep=" "), int(minutes),
                              int(nbhf), int(nbmf), int(other), int(count),
                              ici_str, freq_str, risk_str])
            rows_written += 1

    interval_label = "daily" if args.interval == "day" else "hourly"
    print(f"Wrote {rows_written} {interval_label} rows to "
          f"{args.output} (threshold={args.threshold}); "
          f"{int(presence_minutes.sum())} total positive minutes")


if __name__ == "__main__":
    main()
