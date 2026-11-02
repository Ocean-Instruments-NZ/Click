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
click_s1.py

Combine click detections from a folder of cetacean acoustic detector .bcl
log files into a single .npz file, ordered chronologically using the
timestamp embedded in each filename (...YYMMDDHHMMSS.bcl).

The output file has three arrays, one row per detection:
    t     - datetime64[us] (rtime + mticks/1e6, converted from Unix
            seconds), true UTC, microsecond precision. Note: the .bcl
            filenames are timestamped in local time instead (~12h offset,
            NZ standard time) - don't confuse the two. Downstream modules
            that need numeric arithmetic on it (e.g. click_s2.py's/
            click_s2_b.py's Rayleigh test) convert it back to float64 Unix
            epoch seconds via datetime64_to_epoch_seconds().
    level - float64 noise level in centibels at time of detection
    mf    - float64 "mid frequency" in Hz, computed via FFT on the
            detection's audio snippet from the matching .dwv file (same
            basename as the .bcl, one fixed-length snippet per saved
            detection, concatenated; length set by --snippet-len). The
            frequency of the peak-power bin.
            NaN where no snippet is available (missing .dwv, snippet not
            saved for that detection, or a snippet-count mismatch for that
            file).

All detections are accumulated in memory and written in a single np.savez()
call at the end - the full dataset is only a few MB, so this is simpler
than streaming, unlike the .npz format itself, which doesn't support the
same kind of incremental/chunked append a flat binary file would.

Usage:
    python click_s1.py input_folder [-o output.npz] [--max-files N]
                       [--snippet-len N]

Requires: pandas, numpy
"""

import argparse
import glob
import os
import re
import sys
import wave
from datetime import datetime, timezone

import numpy as np
import pandas as pd

COLUMNS = ["rtime", "mticks", "report", "state", "nl", "thr", "scnt"]

# In-memory record layout: microsecond-resolution datetime64 t, plus
# little-endian float64 level and mid frequency.
DTYPE = np.dtype([("t", "datetime64[us]"), ("level", "<f8"), ("mf", "<f8")])

# Each .dwv snippet is a fixed-length window of audio around one detection.
# Default length in samples; override with --snippet-len.
DEFAULT_SNIPPET_LEN = 864

FNAME_TS_RE = re.compile(r"\.(\d{12})\.bcl$", re.IGNORECASE)


def parse_filename_timestamp(path):
    """Extract the YYMMDDHHMMSS timestamp embedded in a .bcl filename and
    return it as a datetime, or None if the filename doesn't match."""
    match = FNAME_TS_RE.search(os.path.basename(path))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%y%m%d%H%M%S")
    except ValueError:
        return None


def find_sorted_files(input_folder):
    """Glob the folder's *.bcl files and sort them chronologically by the
    timestamp embedded in their filename. Files whose name doesn't match
    the expected pattern are skipped with a warning."""
    paths = glob.glob(os.path.join(input_folder, "*.bcl"))
    if not paths:
        sys.exit(f"No *.bcl files found in {input_folder}")

    dated = []
    for p in paths:
        ts = parse_filename_timestamp(p)
        if ts is None:
            print(f"  skipping {os.path.basename(p)}: filename timestamp "
                  f"not recognized", file=sys.stderr)
            continue
        dated.append((ts, p))

    if not dated:
        sys.exit("No files with a recognizable filename timestamp found.")

    dated.sort(key=lambda pair: pair[0])
    return [p for _, p in dated]


def read_snippets(dwv_path, snippet_len):
    """Read all fixed-length audio snippets from a .dwv file. Returns
    (sample_rate, snippets) where snippets has shape (n_snippets,
    snippet_len), or (None, None) if the file is missing, unreadable, or
    its sample count isn't a whole multiple of snippet_len (better to skip
    than risk misaligned snippets)."""
    if not os.path.exists(dwv_path):
        return None, None
    try:
        with wave.open(dwv_path, "rb") as w:
            sample_rate = w.getframerate()
            raw = w.readframes(w.getnframes())
        samples = np.frombuffer(raw, dtype="<i2")
        if samples.size == 0 or samples.size % snippet_len != 0:
            return None, None
        return sample_rate, samples.reshape(-1, snippet_len)
    except (wave.Error, OSError):
        return None, None


def compute_mf(snippets, sample_rate):
    """Vectorized "mid frequency" (Hz) for every snippet at once via FFT:
    the frequency of the highest-power bin. The DC bin (0 Hz) is excluded,
    since a click's spectral content is never there and any DC offset in
    the raw samples would otherwise dominate the result."""
    freqs = np.fft.rfftfreq(snippets.shape[1], d=1.0 / sample_rate)
    power = np.abs(np.fft.rfft(snippets, axis=1)) ** 2
    freqs, power = freqs[1:], power[:, 1:]
    return freqs[np.argmax(power, axis=1)]


def read_detections(path, snippet_len=DEFAULT_SNIPPET_LEN, stats=None):
    """Read a single .bcl file and return (time, level, mf) float64 arrays
    for its detection ('D') rows only. `mf` is computed via FFT on the
    matching .dwv file's `snippet_len`-sample audio snippets (NaN where
    unavailable). `stats`,
    if given, is a dict of counters updated in place for a summary report
    (missing_dwv, mismatched_dwv) rather than printing per file."""
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skiprows=1,
        na_values=[""],
        dtype={"report": str},
    )
    if df.empty or df["rtime"].isna().all():
        df = pd.read_csv(path, header=None, names=COLUMNS, na_values=[""],
                          dtype={"report": str})

    det = df[df["report"] == "D"]
    if det.empty:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty

    time = (det["rtime"].to_numpy(dtype=np.float64)
            + det["mticks"].to_numpy(dtype=np.float64) / 1e6)
    level = det["nl"].to_numpy(dtype=np.float64)

    saved = det["state"].to_numpy() == 1
    mf = np.full(len(det), np.nan, dtype=np.float64)

    dwv_path = os.path.splitext(path)[0] + ".dwv"
    sample_rate, snippets = read_snippets(dwv_path, snippet_len)
    if snippets is None:
        if stats is not None:
            stats["missing_dwv"] += 1
    elif snippets.shape[0] != saved.sum():
        if stats is not None:
            stats["mismatched_dwv"] += 1
    elif snippets.shape[0] > 0:
        mf[saved] = compute_mf(snippets, sample_rate)

    return time, level, mf


def load_clicks(path):
    """Read back a combined clicks file written by this script into a
    structured numpy array with 't' (datetime64[us]), 'level', and 'mf'
    fields."""
    with np.load(path) as d:
        t, level, mf = d["t"], d["level"], d["mf"]
    arr = np.empty(t.shape[0], dtype=DTYPE)
    arr["t"] = t
    arr["level"] = level
    arr["mf"] = mf
    return arr


def epoch_seconds_to_datetime64(seconds):
    """Convert float64 Unix epoch seconds (fractional/microsecond part
    allowed) to a datetime64[us] array."""
    return (np.round(np.asarray(seconds, dtype=np.float64) * 1_000_000)
            .astype(np.int64).astype("datetime64[us]"))


def datetime64_to_epoch_seconds(t):
    """Convert a datetime64[us] array (as stored in click_s1.npz's 't'
    field) back to float64 Unix epoch seconds, for numeric/arithmetic use
    downstream (e.g. click_s2.py's/click_s2_b.py's Rayleigh test)."""
    return t.astype("datetime64[us]").astype(np.int64) / 1_000_000.0


def convert(input_folder, output, max_files=None,
            snippet_len=DEFAULT_SNIPPET_LEN):
    files = find_sorted_files(input_folder)
    print(f"Found {len(files)} files with recognizable timestamps.")

    if max_files is not None and len(files) > max_files:
        print(f"Limiting to the first {max_files} files (chronologically).")
        files = files[:max_files]

    total_detections = 0
    first_time = None
    last_time = None
    all_times = []
    all_levels = []
    all_mfs = []
    stats = {"missing_dwv": 0, "mismatched_dwv": 0}

    for i, path in enumerate(files, 1):
        try:
            time, level, mf = read_detections(path, snippet_len, stats)
        except Exception as exc:
            print(f"  skipping {os.path.basename(path)}: {exc}",
                  file=sys.stderr)
            continue

        if time.size:
            all_times.append(time)
            all_levels.append(level)
            all_mfs.append(mf)
            total_detections += time.size
            if first_time is None:
                first_time = time[0]
            last_time = time[-1]

        if i % 500 == 0 or i == len(files):
            print(f"  processed {i}/{len(files)} files, "
                  f"{total_detections} detections so far")

    times = np.concatenate(all_times) if all_times else np.empty(0)
    levels = np.concatenate(all_levels) if all_levels else np.empty(0)
    mfs = np.concatenate(all_mfs) if all_mfs else np.empty(0)
    # Write via an open file handle (not a bare path) so numpy doesn't
    # append its default .npz suffix if `output` doesn't already have one.
    t = epoch_seconds_to_datetime64(times)
    with open(output, "wb") as f:
        np.savez(f, t=t, level=levels, mf=mfs)

    print(f"Wrote {total_detections} detections to {output} "
          f"({os.path.getsize(output)} bytes)")
    if first_time is not None:
        print(f"Time span (UTC): "
              f"{datetime.fromtimestamp(first_time, timezone.utc)} "
              f"to {datetime.fromtimestamp(last_time, timezone.utc)}")
    if stats["missing_dwv"] or stats["mismatched_dwv"]:
        print(f"mf unavailable for {stats['missing_dwv']} files with no "
              f".dwv and {stats['mismatched_dwv']} files with a snippet "
              f"count mismatch (mf left as NaN for their detections)")


def main():
    parser = argparse.ArgumentParser(
        description="Combine .bcl click detections into a single .npz "
                     "file, ordered chronologically by filename timestamp."
    )
    parser.add_argument("input_folder",
                         help="Folder containing .bcl files (required)")
    parser.add_argument("-o", "--output", default="click_s1.npz",
                         help="Output .npz file (default: click_s1.npz)")
    parser.add_argument("--max-files", type=int, default=None,
                         help="Maximum number of input files to process, "
                              "in chronological order (default: unlimited)")
    parser.add_argument("--snippet-len", type=int,
                         default=DEFAULT_SNIPPET_LEN,
                         help="Length in samples of each audio snippet in "
                              "the .dwv files (default: "
                              f"{DEFAULT_SNIPPET_LEN})")
    args = parser.parse_args()
    if args.snippet_len < 1:
        parser.error("--snippet-len must be a positive integer")

    import time as time_module
    t0 = time_module.time()
    convert(args.input_folder, args.output, args.max_files, args.snippet_len)
    print(f"Done in {time_module.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
