#!/usr/bin/env python3
"""
click_s2_plot_results.py

Render the periodicity grid and noise-level line produced by click_s2.py
(a click_s2.npz file) as an interactive Plotly HTML heatmap, with a
noise-level line graph sharing the same time axis.

Note: click times are Unix epoch seconds in true UTC. The .bcl filenames
are timestamped in local time instead (~12h offset, NZ standard time) -
so click-to-copy reads each file's actual first record rather than
trusting the filename, to stay correct without hardcoding a timezone.

Clicking a column in the heatmap copies the filename of the first .bcl
file that contributed to it to the clipboard (looked up from --bcl-folder
by reading each file's actual first record time). Pass --bcl-folder '' to
disable.

Usage:
    python click_s2_plot_results.py [click_s2.npz]
        [--plot-output click_periodicity.html] [--plot-decimate N]
        [--bcl-folder Hydrophone] [--start-day N] [--end-day N]

Requires: numpy, plotly
"""

import argparse
import glob
import os
import warnings
from datetime import datetime, timedelta, timezone

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from click_s1 import parse_filename_timestamp

# Above this many time columns, an un-decimated plot risks becoming huge
# enough that browsers fail to render it usefully (a 30-day run at the 2s
# default interval produced a 194 MB HTML that rendered blank).
MAX_PLOT_COLUMNS = 5000


def load_periodicity_data(path):
    """Load a click_s2.npz file, upcasting the float16 arrays to float64
    (numpy math on float16 is slow and imprecise) and reconstructing the
    deterministic interval_start time axis."""
    with np.load(path) as d:
        Z = d["z"].astype(np.float64)
        noise = d["noise_level"].astype(np.float64)
        periods_ms = d["periods_ms"].astype(np.float64)
        t0 = float(d["t0"])
        interval_sec = float(d["interval_sec"])

    interval_start = t0 + np.arange(Z.shape[0]) * interval_sec
    return Z, noise, periods_ms, interval_start


def build_bcl_manifest(folder, pattern="*.bcl"):
    """Scan `folder` for .bcl files and return (start_times, filenames),
    sorted by each file's *actual* first record time (its rtime/mticks
    columns, true UTC) - not the filename timestamp, which is in local
    time (~12h offset, NZ standard time).

    Only reads each file's first data line, so this is cheap per file, but
    opening thousands of small files is still slow (tens of seconds for
    ~6000 files) - the result is cached alongside the folder, keyed by file
    count, and reused on later runs unless files were added/removed.

    Returns (None, None) if the folder/pattern has no usable .bcl files -
    callers should treat that as "click-to-copy disabled" rather than a
    hard error.
    """
    paths = glob.glob(os.path.join(folder, pattern))
    if not paths:
        print(f"Note: no files matching '{pattern}' found in {folder}; "
              f"click-to-copy source filename will be disabled.")
        return None, None

    cache_path = os.path.join(folder, ".bcl_manifest_cache.npz")
    if os.path.exists(cache_path):
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if int(cached["file_count"]) == len(paths):
                print(f"Using cached .bcl manifest ({len(paths)} files) "
                      f"from {cache_path}")
                return cached["starts"], cached["names"].astype(object)
        except (OSError, KeyError, ValueError):
            pass  # fall through and rebuild

    # Sort by filename timestamp first just to process in roughly
    # chronological order (order doesn't matter for correctness below,
    # since we sort by true start time at the end anyway).
    paths.sort(key=lambda p: parse_filename_timestamp(p) or datetime.min)

    starts = []
    names = []
    for p in paths:
        try:
            with open(p) as f:
                next(f)  # header
                line = next(f)
            fields = line.split(",")
            start = float(fields[0]) + float(fields[1]) / 1e6
        except (StopIteration, ValueError, IndexError, OSError):
            continue
        starts.append(start)
        names.append(os.path.basename(p))

    if not starts:
        print(f"Note: could not read a start time from any file in "
              f"{folder}; click-to-copy source filename will be disabled.")
        return None, None

    starts = np.array(starts, dtype=np.float64)
    names = np.array(names, dtype=object)
    order = np.argsort(starts)
    starts, names = starts[order], names[order]

    try:
        np.savez(cache_path, starts=starts, names=names.astype(str),
                  file_count=len(paths))
    except OSError:
        pass  # non-fatal - just means no caching this run

    return starts, names


def lookup_source_files(column_times, manifest_times, manifest_names):
    """For each column start time, return the basename of the .bcl file
    that was active at (i.e. started most recently at or before) that
    time - the "first" file contributing to that column."""
    idx = np.searchsorted(manifest_times, column_times, side="right") - 1
    idx = np.clip(idx, 0, len(manifest_times) - 1)
    return manifest_names[idx]


def decimate_for_plot(Z, interval_start, factor, agg=np.nanmax):
    """Aggregate every `factor` consecutive interval columns of Z into one,
    via `agg` (default nanmax, which preserves periodicity peaks better
    than averaging; pass np.nanmean for e.g. a noise-level line). Only
    affects plotting - the caller's full-resolution Z/interval_start are
    left untouched. factor <= 1 returns the inputs unchanged.

    Z may be 1D (e.g. noise levels) or 2D (Z-score grid, columns=intervals).
    """
    if factor <= 1:
        return Z, interval_start

    was_1d = Z.ndim == 1
    if was_1d:
        Z = Z[:, np.newaxis]

    n_intervals = Z.shape[0]
    n_groups = -(-n_intervals // factor)  # ceil division
    pad = n_groups * factor - n_intervals
    if pad:
        Z = np.vstack([Z, np.full((pad, Z.shape[1]), np.nan)])

    Z_grouped = Z.reshape(n_groups, factor, Z.shape[1])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # All-NaN groups (empty intervals) -> nanmax/nanmean both return
        # NaN, which is correct; numpy's "All-NaN slice"/"Mean of empty
        # slice" warnings for this are expected noise here.
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        Z_plot = agg(Z_grouped, axis=1)
    interval_start_plot = interval_start[::factor]
    if was_1d:
        Z_plot = Z_plot[:, 0]
    return Z_plot, interval_start_plot


def plot_heatmap_html(Z, periods_ms, interval_start, output_path,
                       noise_level, color_range=None, source_files=None):
    x = np.array(interval_start * 1e9, dtype="datetime64[ns]")

    if color_range is not None:
        z_min, z_max = color_range
    else:
        z_min = float(np.nanmin(Z))
        z_max = float(np.nanmax(Z))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min >= z_max:
        z_min, z_max = 0.0, 1.0
    step = (z_max - z_min) / 200.0

    # One source .bcl filename per column, repeated down every row, so a
    # click on any cell in that column can read it straight off the point.
    customdata = None
    if source_files is not None:
        customdata = np.tile(np.asarray(source_files), (periods_ms.size, 1))

    heatmap = go.Heatmap(
        x=x,
        y=periods_ms,
        z=Z.T,
        customdata=customdata,
        colorscale="Viridis",
        zmin=z_min,
        zmax=z_max,
        colorbar=dict(title="Rayleigh Z", y=0.85, len=0.7),
    )

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.75, 0.25], vertical_spacing=0.05)
    fig.add_trace(heatmap, row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=noise_level, mode="lines",
                              name="Noise level (cB)",
                              line=dict(color="firebrick", width=1)),
                   row=2, col=1)
    fig.update_yaxes(title_text="Period (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Noise level (cB)", row=2, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)

    fig.update_layout(
        title="Click periodicity (Rayleigh Z score)",
        template="plotly_white",
    )

    # Min/max range sliders that restyle the heatmap's color scale live,
    # since Plotly has no built-in colorbar range control.
    div_id = "click-heatmap"
    color_scale_js = """
    var plotDiv = document.getElementById('%s');
    var dataMin = %s;
    var dataMax = %s;
    var sliderStep = %s;

    var controls = document.createElement('div');
    controls.style.cssText = 'display:flex;flex-wrap:wrap;gap:16px;' +
        'align-items:center;margin:8px 4px;font-family:sans-serif;' +
        'font-size:13px;';

    function makeSlider(labelText, initial) {
        var label = document.createElement('label');
        label.textContent = labelText;
        var slider = document.createElement('input');
        slider.type = 'range';
        slider.min = dataMin;
        slider.max = dataMax;
        slider.step = sliderStep;
        slider.value = initial;
        slider.style.verticalAlign = 'middle';
        var valSpan = document.createElement('span');
        valSpan.textContent = initial.toFixed(2);
        valSpan.style.marginLeft = '4px';
        label.appendChild(slider);
        label.appendChild(valSpan);
        return {label: label, slider: slider, valSpan: valSpan};
    }

    var minCtl = makeSlider('Color min: ', dataMin);
    var maxCtl = makeSlider('Color max: ', dataMax);
    var resetBtn = document.createElement('button');
    resetBtn.textContent = 'Reset';

    controls.appendChild(minCtl.label);
    controls.appendChild(maxCtl.label);
    controls.appendChild(resetBtn);
    plotDiv.parentNode.insertBefore(controls, plotDiv);

    function updateColorScale() {
        var zmin = parseFloat(minCtl.slider.value);
        var zmax = parseFloat(maxCtl.slider.value);
        if (zmin >= zmax) { return; }
        minCtl.valSpan.textContent = zmin.toFixed(2);
        maxCtl.valSpan.textContent = zmax.toFixed(2);
        Plotly.restyle(plotDiv, {zmin: [zmin], zmax: [zmax]}, [0]);
    }

    minCtl.slider.addEventListener('input', updateColorScale);
    maxCtl.slider.addEventListener('input', updateColorScale);
    resetBtn.addEventListener('click', function() {
        minCtl.slider.value = dataMin;
        maxCtl.slider.value = dataMax;
        updateColorScale();
    });
    """ % (div_id, f"{z_min:.6g}", f"{z_max:.6g}", f"{step:.6g}")

    click_to_copy_js = ""
    if source_files is not None:
        click_to_copy_js = """
        var toast = document.createElement('div');
        toast.style.cssText = 'position:fixed;bottom:24px;left:50%;' +
            'transform:translateX(-50%);background:#333;color:#fff;' +
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
                showToast('Could not copy - ' + text);
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
        """

    fig.write_html(output_path, include_plotlyjs="cdn", div_id=div_id,
                    post_script=color_scale_js + click_to_copy_js)
    print(f"Saved heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot the periodicity heatmap and noise-level line "
                     "from a click_s2.npz file produced by click_s2.py."
    )
    parser.add_argument("input", nargs="?", default="click_s2.npz",
                         help="Periodicity data file (default: "
                              "click_s2.npz)")
    parser.add_argument("--plot-output", default="click_periodicity.html",
                         help="Heatmap output path (default: "
                              "click_periodicity.html)")
    parser.add_argument("--plot-decimate", type=int, default=1,
                         help="Aggregate every N time intervals into one "
                              "for the plot (default: 1, no decimation, "
                              "unless auto-decimation kicks in - see "
                              "MAX_PLOT_COLUMNS)")
    parser.add_argument("--bcl-folder", default="Hydrophone",
                         help="Folder of source .bcl files, used to enable "
                              "click-to-copy: clicking a column in the "
                              "heatmap copies the filename of the .bcl "
                              "file that contributed to it (default: "
                              "Hydrophone; pass '' to disable)")
    parser.add_argument("--start-day", type=int, default=None,
                         help="Limit the plot to start at this day of year "
                              "(1-365, based on the calendar year of the "
                              "data's first interval)")
    parser.add_argument("--end-day", type=int, default=None,
                         help="Limit the plot to end at this day of year "
                              "(1-365, inclusive)")
    args = parser.parse_args()

    Z, noise, periods_ms, interval_start = load_periodicity_data(args.input)
    print(f"Loaded {Z.shape[0]} intervals x {Z.shape[1]} periods from "
          f"{args.input}")

    if args.start_day is not None or args.end_day is not None:
        for name, day in (("--start-day", args.start_day),
                           ("--end-day", args.end_day)):
            if day is not None and not (1 <= day <= 365):
                raise SystemExit(f"{name} must be between 1 and 365")

        year = datetime.fromtimestamp(interval_start[0], timezone.utc).year
        jan1 = datetime(year, 1, 1, tzinfo=timezone.utc)
        start_ts = ((jan1 + timedelta(days=args.start_day - 1)).timestamp()
                    if args.start_day is not None else -np.inf)
        end_ts = ((jan1 + timedelta(days=args.end_day)).timestamp()
                  if args.end_day is not None else np.inf)

        mask = (interval_start >= start_ts) & (interval_start < end_ts)
        if not mask.any():
            raise SystemExit(
                "No data remains after applying --start-day/--end-day")
        Z, noise, interval_start = Z[mask], noise[mask], interval_start[mask]
        print(f"Limited to day {args.start_day or 1}-"
              f"{args.end_day or 365} of {year}: {Z.shape[0]} intervals")

    # Full-resolution range, captured before any plot-only decimation, so
    # the color slider always spans the true data range rather than the
    # min-of-decimated-maxes.
    full_range = (float(np.nanmin(Z)), float(np.nanmax(Z)))

    decimate = args.plot_decimate
    if decimate <= 1 and Z.shape[0] > MAX_PLOT_COLUMNS:
        decimate = -(-Z.shape[0] // MAX_PLOT_COLUMNS)  # ceil division
        print(f"Note: {Z.shape[0]} time columns is too many to plot "
              f"safely; auto-decimating by {decimate}x for the plot only. "
              f"Pass --plot-decimate to control this explicitly.")

    Z_plot, interval_start_plot = decimate_for_plot(Z, interval_start,
                                                      decimate)
    # Same decimate factor as Z, so the two traces stay column-aligned.
    noise_plot, _ = decimate_for_plot(noise, interval_start, decimate,
                                       agg=np.nanmean)

    source_files = None
    if args.bcl_folder:
        manifest_times, manifest_names = build_bcl_manifest(args.bcl_folder)
        if manifest_times is not None:
            source_files = lookup_source_files(
                interval_start_plot, manifest_times, manifest_names)

    plot_heatmap_html(Z_plot, periods_ms, interval_start_plot,
                       args.plot_output, noise_plot, color_range=full_range,
                       source_files=source_files)


if __name__ == "__main__":
    main()
