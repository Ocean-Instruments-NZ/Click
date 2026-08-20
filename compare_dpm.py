#!/usr/bin/env python3
"""
compare_dpm.py

Compare two detection-positive-minutes (DPM) time series - the
SoundTrap/click_periodicity pipeline's output and an independent CPOD
hydrophone's output - and report the Pearson correlation and Spearman's
rank correlation as text. Optionally produce a scatter plot (PNG) with a
least-squares linear fit line and its equation. Accepts either daily or
hourly data for both files - granularity is auto-detected from each
file's own content, and the two files must match (see "Compatibility
check" below).

Input formats:
    SoundTrap (click_s3.csv / clicks_dpm.csv): header row, then
        datetime_utc,cetacean_present_minutes[,...]
        daily:  2017-06-27 00:00:00+00:00,7
        hourly: 2017-06-27 08:00:00+00:00,2
    Only the first two columns are used, so extra columns (click_s3.csv's
    ici/frequency etc.) are ignored.

    CPOD, daily (e.g. CPOD.csv): no header,
        DD/MM/YY,minutes
        27/06/17,0

    CPOD, hourly (e.g. CPOD Hourly.csv): header row "ChunkEnd,DPM", then
        DD/MM/YY HH:MM,minutes
        28/06/17 08:59,0
    (the hour is read from the chunk's end-of-hour timestamp - "08:59"
    and click_s3.py's hour-start "08:00:00" both refer to the same
    08:00-08:59 window, so they join correctly without adjustment)

Granularity is auto-detected per file (repeated dates or a non-midnight
time of day means hourly) - there's no CLI flag for it. Timestamps are
used as given, with no timezone adjustment applied by this script (the
same as today's daily CPOD.csv, which is expected to already be
UTC-shifted by whatever exported it).

Compatibility check: both files must be the same granularity (both daily
or both hourly) - a mismatch raises a clear error before any join is
attempted, rather than silently producing an empty or wrong comparison.
Pass --aggregate-daily to resolve a mismatch (or to force a daily
comparison from two hourly files) by summing each hourly file's rows into
daily totals before the compatibility check and join - an already-daily
file is left unchanged. Partial days (e.g. a deployment's first/last day)
are summed from whatever hourly rows are present, with no attempt to
fabricate missing hours.

Usage:
    python compare_dpm.py [click_s3.csv] [CPOD.csv]
        [--aggregate-daily] [-o report.txt] [--plot-output scatter.png]

Requires: numpy, scipy, matplotlib (only if --plot-output is given)
"""

import argparse
import csv
from datetime import datetime

import numpy as np
from scipy import stats


def load_soundtrap_csv(path):
    """Return (values, granularity) from a click_s3.csv (or
    clicks_dpm.csv) file. granularity is "hourly" if any date repeats
    across rows or any row's time-of-day isn't midnight, else "daily".
    values keys are date objects for daily files, (date, hour) tuples for
    hourly ones. Raises if the same key appears twice - a real data
    problem, not just a granularity signal."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            rows.append((datetime.fromisoformat(row[0]), int(row[1])))

    dates = [ts.date() for ts, _ in rows]
    is_hourly = (len(set(dates)) < len(dates)
                 or any(ts.hour or ts.minute for ts, _ in rows))

    values = {}
    for ts, minutes in rows:
        key = (ts.date(), ts.hour) if is_hourly else ts.date()
        if key in values:
            raise SystemExit(
                f"{path} has more than one row for {format_key(key)} - "
                f"check the file for duplicate or corrupted rows.")
        values[key] = minutes
    return values, ("hourly" if is_hourly else "daily")


def load_cpod_csv(path):
    """Return (values, granularity) from a CPOD.csv-style file - daily
    (no header, DD/MM/YY,minutes) or hourly (header "ChunkEnd,DPM",
    DD/MM/YY HH:MM,minutes). The header row (if present) is detected by
    trying to parse the first field as a date and treating a failure as a
    header, rather than assuming a fixed format. values keys are date
    objects for daily files, (date, hour) tuples for hourly ones."""
    with open(path, newline="") as f:
        rows = [row for row in csv.reader(f) if row]

    if rows:
        try:
            datetime.strptime(rows[0][0].split(" ")[0], "%d/%m/%y")
        except ValueError:
            rows = rows[1:]  # header row (e.g. "ChunkEnd,DPM")

    is_hourly = any(" " in row[0] for row in rows)

    values = {}
    for row in rows:
        if is_hourly:
            ts = datetime.strptime(row[0], "%d/%m/%y %H:%M")
            key = (ts.date(), ts.hour)
        else:
            key = datetime.strptime(row[0], "%d/%m/%y").date()
        if key in values:
            raise SystemExit(
                f"{path} has more than one row for {format_key(key)} - "
                f"check the file for duplicate or corrupted rows.")
        values[key] = int(row[1])
    return values, ("hourly" if is_hourly else "daily")


def aggregate_to_daily(values, granularity):
    """If `values` is hourly (keyed by (date, hour)), sum each date's
    hourly minutes into a single daily total, keyed by date. No-op
    (returned unchanged) if already daily. Partial days (fewer than 24
    hourly rows present) are summed from whatever rows exist - no attempt
    to fabricate missing hours."""
    if granularity == "daily":
        return values, "daily"
    daily = {}
    for (date, hour), minutes in values.items():
        daily[date] = daily.get(date, 0) + minutes
    return daily, "daily"


def check_compatible_granularity(soundtrap_gran, cpod_gran,
                                  soundtrap_path, cpod_path):
    """Raise SystemExit if the two files aren't the same granularity -
    joining an hourly file against a daily one would silently produce an
    empty or misleading comparison, so this is checked explicitly before
    any join is attempted."""
    if soundtrap_gran != cpod_gran:
        raise SystemExit(
            f"Time interval mismatch: {soundtrap_path} is {soundtrap_gran} "
            f"but {cpod_path} is {cpod_gran}. Both files must be the same "
            f"granularity - use matching hourly or daily files for both, "
            f"or pass --aggregate-daily to sum hourly data to daily "
            f"totals before comparing.")


def format_key(key):
    """Render a join key (a date, or a (date, hour) tuple) for display."""
    if isinstance(key, tuple):
        date, hour = key
        return f"{date} {hour:02d}:00"
    return str(key)


def join_series(soundtrap, cpod):
    """Inner join on the shared key type (date, or (date, hour) for
    hourly data), sorted. Returns (keys, soundtrap_vals, cpod_vals)."""
    common = sorted(set(soundtrap) & set(cpod))
    if not common:
        raise SystemExit("No overlapping time periods between the two files.")
    soundtrap_only = len(soundtrap) - len(common)
    cpod_only = len(cpod) - len(common)
    print(f"Matched {len(common)} periods "
          f"({soundtrap_only} only in SoundTrap file, "
          f"{cpod_only} only in CPOD file)")
    soundtrap_vals = np.array([soundtrap[k] for k in common], dtype=np.float64)
    cpod_vals = np.array([cpod[k] for k in common], dtype=np.float64)
    return common, soundtrap_vals, cpod_vals


def plot_scatter(cpod_vals, soundtrap_vals, unit, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    slope, intercept, r_value, _, _ = stats.linregress(cpod_vals, soundtrap_vals)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cpod_vals, soundtrap_vals, alpha=0.6, edgecolor="k")

    x_line = np.linspace(cpod_vals.min(), cpod_vals.max(), 100)
    y_line = slope * x_line + intercept
    sign = "+" if intercept >= 0 else "-"
    label = (f"y = {slope:.3f}x {sign} {abs(intercept):.3f}\n"
             f"R² = {r_value ** 2:.3f}")
    ax.plot(x_line, y_line, color="firebrick", label=label)

    ax.set_xlabel(f"CPOD detection-positive minutes/{unit}")
    ax.set_ylabel(f"SoundTrap detection-positive minutes/{unit}")
    ax.set_title("SoundTrap vs CPOD detection-positive minutes")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SoundTrap and CPOD detection-positive-minutes "
                     "time series (Pearson correlation and Spearman's "
                     "rank correlation, optional scatter plot). Accepts "
                     "hourly or daily data for both files - granularity "
                     "is auto-detected and must match."
    )
    parser.add_argument("soundtrap_csv", nargs="?",
                         default="click_s3.csv",
                         help="SoundTrap DPM CSV (default: click_s3.csv)")
    parser.add_argument("cpod_csv", nargs="?", default="CPOD Data/CPOD Hourly.csv",
                         help="CPOD DPM CSV (default: CPOD Data/CPOD.csv)")
    parser.add_argument("--aggregate-daily", action="store_true",
                         help="Sum any hourly input file into daily "
                              "totals before comparing (resolves an "
                              "hourly/daily mismatch, or forces a daily "
                              "comparison from two hourly files; "
                              "default: off, use each file's granularity "
                              "as loaded)")
    parser.add_argument("-o", "--output", default=None,
                         help="Also write the text report to this path "
                              "(default: print only)")
    parser.add_argument("--plot-output", default=None,
                         help="Save a scatter plot with a least-squares "
                              "fit line to this PNG path (default: no plot)")
    args = parser.parse_args()

    soundtrap, soundtrap_gran = load_soundtrap_csv(args.soundtrap_csv)
    cpod, cpod_gran = load_cpod_csv(args.cpod_csv)
    soundtrap_gran_orig, cpod_gran_orig = soundtrap_gran, cpod_gran

    if args.aggregate_daily:
        soundtrap, soundtrap_gran = aggregate_to_daily(soundtrap, soundtrap_gran)
        cpod, cpod_gran = aggregate_to_daily(cpod, cpod_gran)

    check_compatible_granularity(soundtrap_gran, cpod_gran,
                                  args.soundtrap_csv, args.cpod_csv)
    unit = "hour" if soundtrap_gran == "hourly" else "day"

    keys, soundtrap_vals, cpod_vals = join_series(soundtrap, cpod)

    pearson_r, pearson_p = stats.pearsonr(soundtrap_vals, cpod_vals)
    spearman_r, spearman_p = stats.spearmanr(soundtrap_vals, cpod_vals)

    aggregated_notes = []
    if soundtrap_gran_orig == "hourly" and soundtrap_gran == "daily":
        aggregated_notes.append("SoundTrap aggregated from hourly")
    if cpod_gran_orig == "hourly" and cpod_gran == "daily":
        aggregated_notes.append("CPOD aggregated from hourly")
    granularity_line = soundtrap_gran
    if aggregated_notes:
        granularity_line += f" ({'; '.join(aggregated_notes)})"

    report = (
        f"SoundTrap file:    {args.soundtrap_csv}\n"
        f"CPOD file:         {args.cpod_csv}\n"
        f"Granularity:       {granularity_line}\n"
        f"Matched {unit}s: {len(keys)} "
        f"({format_key(keys[0])} to {format_key(keys[-1])})\n"
        f"Pearson r:         {pearson_r:.4f} (p={pearson_p:.4g})\n"
        f"Spearman rank r:   {spearman_r:.4f} (p={spearman_p:.4g})\n"
    )
    print(report, end="")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Wrote report to {args.output}")

    if args.plot_output:
        plot_scatter(cpod_vals, soundtrap_vals, unit, args.plot_output)


if __name__ == "__main__":
    main()
