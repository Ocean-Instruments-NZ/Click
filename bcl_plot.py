#!/usr/bin/env python3
"""
bcl_plot.py

Read a folder of cetacean acoustic detector .bcl log files and produce an
interactive plot (zoomable/pannable) of detections binned per hour.

Usage:
    python bcl_plot.py /path/to/folder [-o output.html] [--pattern "*.bcl"]

Requires: pandas, plotly  (pip install pandas plotly)
"""

import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import plotly.graph_objects as go


COLUMNS = ["rtime", "mticks", "report", "state", "nl", "thr", "scnt"]


def read_bcl_file(path):
    """Read a single .bcl file into a DataFrame with a proper timestamp."""
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skiprows=1,          # first line is a quoted header, e.g. "rtime","mticks",...
        na_values=[""],
        dtype={"report": str},  # state/nl left as float64 (faster than Int64)
    )
    # Guard against a file that has no header row: if the first row's
    # "rtime" isn't numeric, pandas will have already dropped it via
    # skiprows=1 assuming it *is* a header. If the file lacks a header,
    # re-read without skipping.
    if df.empty or df["rtime"].isna().all():
        df = pd.read_csv(path, header=None, names=COLUMNS, na_values=[""],
                          dtype={"report": str})

    df["timestamp"] = pd.to_datetime(
        df["rtime"] + df["mticks"] / 1e6, unit="s", utc=True
    )
    df["source_file"] = os.path.basename(path)
    return df


def load_folder(folder, pattern="*.bcl", max_files=200, workers=8):
    """Read and concatenate the first `max_files` matching files (sorted by
    filename) in a folder, using a thread pool since reading is I/O-bound."""
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        sys.exit(f"No files matching '{pattern}' found in {folder}")

    if len(paths) > max_files:
        print(f"Found {len(paths)} files; limiting to the first {max_files}.")
        paths = paths[:max_files]

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_path = {pool.submit(read_bcl_file, p): p for p in paths}
        for future in as_completed(future_to_path):
            p = future_to_path[future]
            try:
                frames.append(future.result())
            except Exception as exc:
                print(f"  skipping {os.path.basename(p)}: {exc}", file=sys.stderr)

    if not frames:
        sys.exit("No files could be read successfully.")

    data = pd.concat(frames, ignore_index=True, copy=False)
    data = data.sort_values("timestamp").reset_index(drop=True)
    return data


def get_effort_intervals(data):
    """Pair up on-effort (state=1) / off-effort (state=0) 'E' rows into
    (start, end) intervals, per source file (in case files overlap in time
    or a file has multiple on/off cycles). Used only for console stats.
    Vectorized (no per-row Python loop) for speed on large datasets."""
    effort = data.loc[data["report"] == "E", ["source_file", "timestamp", "state"]]
    effort = effort.sort_values(["source_file", "timestamp"])

    intervals = []
    for _, grp in effort.groupby("source_file", sort=False):
        starts = grp.loc[grp["state"] == 1, "timestamp"].to_numpy()
        ends = grp.loc[grp["state"] == 0, "timestamp"].to_numpy()
        # Pair them positionally in chronological order; if counts differ
        # (unmatched on/off), pair what we can and drop the remainder.
        n = min(len(starts), len(ends))
        intervals.extend(zip(starts[:n], ends[:n]))
        # Any leftover off-effort rows with no matching start still count
        # as zero-width intervals, matching prior behavior.
        if len(ends) > n:
            intervals.extend((e, e) for e in ends[n:])
    return intervals


def plot_detections(data, output_path, title=None):
    detections = data[data["report"] == "D"].copy()
    intervals = get_effort_intervals(data)

    fig = go.Figure()

    if not detections.empty:
        counts = (
            detections
            .set_index("timestamp")
            .resample("1h")
            .size()
        )
        # Fill in any empty hours within the overall time range so bars
        # are evenly spaced and zero-count hours show as zero, not gaps.
        full_range = pd.date_range(counts.index.min(), counts.index.max(),
                                    freq="1h")
        counts = counts.reindex(full_range, fill_value=0)

        # Average noise level per hour, for extra hover context
        mean_nl = (
            detections
            .set_index("timestamp")["nl"]
            .resample("1h")
            .mean()
            .reindex(full_range)
        )

        # First source file (by earliest detection timestamp) in each hour bin
        detections_sorted = detections.sort_values("timestamp")
        first_file = (
            detections_sorted
            .groupby(detections_sorted["timestamp"].dt.floor("1h"))["source_file"]
            .first()
            .reindex(full_range)
        )

        bin_end = counts.index + pd.Timedelta(hours=1)
        hover_text = []
        for start, end, count, nl, fname in zip(
            counts.index, bin_end, counts.values, mean_nl.values, first_file.values
        ):
            line = f"{start:%Y-%m-%d %H:%M} \u2013 {end:%H:%M}<br>Detections: {count}"
            if pd.notna(nl):
                line += f"<br>Avg noise level: {nl:.1f} cB"
            if pd.notna(fname):
                line += f"<br>File: {fname}<br><i>(click bar to copy filename)</i>"
            hover_text.append(line)

        # customdata carries the raw filename (empty string if none) so the
        # click handler below can copy it without re-parsing hover text.
        customdata = [fname if pd.notna(fname) else "" for fname in first_file.values]

        fig.add_trace(go.Bar(
            x=counts.index,
            y=counts.values,
            marker_color="crimson",
            name=f"Detections/hour (n={len(detections)} total)",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            customdata=customdata,
        ))

        print(f"Hourly detection counts: min={counts.min()}, "
              f"max={counts.max()}, mean={counts.mean():.1f}")

    fig.update_layout(
        title=title or "Detections per hour",
        xaxis_title="Time (UTC)",
        yaxis_title="Detections per hour",
        bargap=0.1,
        template="plotly_white",
    )
    # Explicit, zoomable/pannable time axis
    fig.update_xaxes(type="date", rangeslider_visible=False)
    fig.update_yaxes(rangemode="tozero")

    div_id = "bcl-plot"
    click_to_copy_js = """
    var plotDiv = document.getElementById('%s');
    var toast = document.createElement('div');
    toast.style.cssText = 'position:fixed;bottom:24px;left:50%%;' +
        'transform:translateX(-50%%);background:#333;color:#fff;' +
        'padding:8px 16px;border-radius:4px;font-family:sans-serif;' +
        'font-size:14px;opacity:0;transition:opacity 0.3s;z-index:9999;' +
        'pointer-events:none;';
    document.body.appendChild(toast);

    function showToast(msg) {
        toast.textContent = msg;
        toast.style.opacity = '1';
        setTimeout(function() { toast.style.opacity = '0'; }, 1500);
    }

    function copyText(text) {
        if (navigator.clipboard && window.isSecureContext) {
            navigator.clipboard.writeText(text).then(
                function() { showToast('Copied: ' + text); },
                function() { fallbackCopy(text); }
            );
        } else {
            fallbackCopy(text);
        }
    }

    function fallbackCopy(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try {
            document.execCommand('copy');
            showToast('Copied: ' + text);
        } catch (err) {
            showToast('Could not copy \u2014 ' + text);
        }
        document.body.removeChild(ta);
    }

    plotDiv.on('plotly_click', function(data) {
        var point = data.points[0];
        var fname = point.customdata;
        if (fname) {
            copyText(fname);
        }
    });
    """ % div_id

    fig.write_html(output_path, div_id=div_id, post_script=click_to_copy_js)
    print(f"Saved interactive plot to {output_path}")
    print(f"Files processed: {data['source_file'].nunique()}")
    print(f"Total detections: {len(detections)}")
    print(f"Effort periods: {len(intervals)}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot cetacean detections over time from a folder of .bcl files."
    )
    parser.add_argument("folder", help="Folder containing .bcl files")
    parser.add_argument("-o", "--output", default="detections.html",
                         help="Output plot file (default: detections.html)")
    parser.add_argument("--pattern", default="*.bcl",
                         help="Glob pattern for input files (default: *.bcl)")
    parser.add_argument("--max-files", type=int, default=200,
                         help="Maximum number of files to process, sorted by "
                              "filename (default: 200)")
    parser.add_argument("--workers", type=int, default=8,
                         help="Number of files to read in parallel (default: 8)")
    args = parser.parse_args()

    import time
    t0 = time.time()
    data = load_folder(args.folder, args.pattern, args.max_files, args.workers)
    print(f"Loaded {len(data)} rows in {time.time() - t0:.1f}s")
    plot_detections(data, args.output)


if __name__ == "__main__":
    main()
