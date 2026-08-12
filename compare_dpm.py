#!/usr/bin/env python3
"""
compare_dpm.py

Compare two daily detection-positive-minutes (DPM) time series - the
SoundTrap/click_periodicity pipeline's output and an independent CPOD
hydrophone's output - and report Pearson and Spearman correlations as
text. Optionally produce a scatter plot (PNG) with a least-squares linear
fit line and its equation.

Input formats:
    SoundTrap (click_s3.csv): header row, then
        datetime_utc,cetacean_present_minutes[,ici,frequency]
        2017-06-27 00:00:00+00:00,7
    CPOD (CPOD.csv): no header,
        DD/MM/YY,minutes
        27/06/17,0
Both are daily counts; joined on date (CPOD has no time-of-day component).
Only the first two SoundTrap columns are used, so click_s3.csv's extra ici/
frequency columns (and clicks_dpm.csv, which lacks them) both work here.

Usage:
    python compare_dpm.py [click_s3.csv] [CPOD.csv]
        [-o report.txt] [--plot-output scatter.png]

Requires: numpy, scipy, matplotlib (only if --plot-output is given)
"""

import argparse
import csv
from datetime import datetime

import numpy as np
from scipy import stats


def load_soundtrap_csv(path):
    """Return {date: cetacean_present_minutes} from a click_s3.csv (or
    clicks_dpm.csv) file. Raises if any date repeats (e.g. an hourly file
    passed by mistake - regenerate with the default daily --interval)."""
    values = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            date = datetime.fromisoformat(row[0]).date()
            if date in values:
                raise SystemExit(
                    f"{path} has more than one row for {date} - it looks "
                    f"like an hourly file. Regenerate it with click_s3.py's "
                    f"default daily --interval.")
            values[date] = int(row[1])
    return values


def load_cpod_csv(path):
    """Return {date: minutes} from a CPOD.csv file (no header,
    DD/MM/YY,minutes)."""
    values = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            date = datetime.strptime(row[0], "%d/%m/%y").date()
            if date in values:
                raise SystemExit(f"{path} has more than one row for {date}")
            values[date] = int(row[1])
    return values


def join_series(soundtrap, cpod):
    """Inner join on date, sorted. Returns (dates, soundtrap_vals, cpod_vals)."""
    common = sorted(set(soundtrap) & set(cpod))
    if not common:
        raise SystemExit("No overlapping dates between the two files.")
    soundtrap_only = len(soundtrap) - len(common)
    cpod_only = len(cpod) - len(common)
    print(f"Matched {len(common)} days "
          f"({soundtrap_only} only in SoundTrap file, "
          f"{cpod_only} only in CPOD file)")
    soundtrap_vals = np.array([soundtrap[d] for d in common], dtype=np.float64)
    cpod_vals = np.array([cpod[d] for d in common], dtype=np.float64)
    return common, soundtrap_vals, cpod_vals


def plot_scatter(cpod_vals, soundtrap_vals, output_path):
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

    ax.set_xlabel("CPOD detection-positive minutes/day")
    ax.set_ylabel("SoundTrap detection-positive minutes/day")
    ax.set_title("SoundTrap vs CPOD detection-positive minutes")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved scatter plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SoundTrap and CPOD detection-positive-minutes "
                     "time series (Pearson/Spearman correlation, optional "
                     "scatter plot)."
    )
    parser.add_argument("soundtrap_csv", nargs="?",
                         default="click_s3.csv",
                         help="SoundTrap DPM CSV (default: click_s3.csv)")
    parser.add_argument("cpod_csv", nargs="?", default="CPOD.csv",
                         help="CPOD DPM CSV (default: CPOD.csv)")
    parser.add_argument("-o", "--output", default=None,
                         help="Also write the text report to this path "
                              "(default: print only)")
    parser.add_argument("--plot-output", default=None,
                         help="Save a scatter plot with a least-squares "
                              "fit line to this PNG path (default: no plot)")
    args = parser.parse_args()

    soundtrap = load_soundtrap_csv(args.soundtrap_csv)
    cpod = load_cpod_csv(args.cpod_csv)
    dates, soundtrap_vals, cpod_vals = join_series(soundtrap, cpod)

    pearson_r, pearson_p = stats.pearsonr(soundtrap_vals, cpod_vals)
    spearman_r, spearman_p = stats.spearmanr(soundtrap_vals, cpod_vals)

    report = (
        f"SoundTrap file: {args.soundtrap_csv}\n"
        f"CPOD file:      {args.cpod_csv}\n"
        f"Matched days:   {len(dates)} "
        f"({dates[0]} to {dates[-1]})\n"
        f"Pearson r:      {pearson_r:.4f} (p={pearson_p:.4g})\n"
        f"Spearman r:     {spearman_r:.4f} (p={spearman_p:.4g})\n"
    )
    print(report, end="")

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Wrote report to {args.output}")

    if args.plot_output:
        plot_scatter(cpod_vals, soundtrap_vals, args.plot_output)


if __name__ == "__main__":
    main()
