"""Visualise typing EEG formats before and after the typing v2 changes.

Top: one real validated trial with more than 50 keys, showing which EEG each format feeds the model.
Bottom: units per typing->response pair across all trials, with the old (50) and new (100) limits.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from typing_annotations import read_windows, replay, text_matches  # noqa: E402

DATASET = ROOT / "dataset"
PAIRS = DATASET / "authentic_typing_pairs"
SFREQ, PRE, WINDOW = 512.0, 102, 513
OLD_LIMIT, NEW_LIMIT = 50, 100


def example_trial(min_chars=58, max_chars=75):
    manifest = json.loads((DATASET / "cohort_manifest.json").read_text())
    for participant in manifest["participants"]:
        for session in participant.get("sessions", []):
            session_dir = Path(session.get("output_dir", ""))
            log = Path(session.get("selected_psychopy_csv", "")).with_suffix(".log")
            pid = participant["participant_id"]
            segments = session_dir / f"{pid}_typing_segments.csv"
            if not session_dir.name.startswith("ses-") or not log.is_file() or not segments.is_file():
                continue
            windows = read_windows(log)
            rate = json.loads((session_dir / f"{pid}_alignment_report.json").read_text())["a"]
            for row in csv.DictReader(open(segments)):
                text = row["typed_text"].rstrip()
                if not min_chars <= len(text) <= max_chars or "\n" in text:
                    continue
                onset, offset = float(row["onset"]), float(row["offset"])
                matches = [w for w in windows if w["part"] == int(row["part"])
                           and onset - 0.05 <= w["log_start_s"] <= offset]
                if len(matches) != 1:
                    continue
                characters, unsupported, _ = replay(matches[0]["keys"])
                if unsupported or not text_matches(characters, text):
                    continue
                fif = next(path for path in (session_dir / "typing_raw_segments").glob("*.fif")
                           if mne.io.read_raw_fif(path, verbose="ERROR").first_samp == int(row["eeg_start_sample"]))
                return pid, row, text, characters, matches[0], rate, onset, fif
    raise RuntimeError("No suitable example trial found")


def load_example():
    pid, row, text, characters, window, rate, onset, fif = example_trial()
    to_s = lambda t: rate * (t - onset)  # seconds from typing-segment start
    raw = mne.io.read_raw_fif(fif, preload=True, verbose="ERROR")
    trace = raw.get_data(picks=[raw.ch_names.index("Cz") if "Cz" in raw.ch_names else 0])[0] * 1e6
    keys = [(to_s(t), c) for c, t in characters]
    words, current = [], []
    for c, t in characters:
        if c.isspace():
            if current:
                words.append(current)
            current = []
        else:
            current.append(to_s(t))
    if current:
        words.append(current)
    logged = [to_s(e["log_time_s"]) for e in window["keys"]]
    return pid, text, keys, words, logged, trace


def units_per_pair(name):
    pairs = np.load(PAIRS / f"authentic_typing_{name}_query_response_pairs.npy", allow_pickle=True).item()["ict_pairs"]
    return np.array([p["num_eeg_units"] for p in pairs])


def main():
    pid, text, keys, words, logged, trace = load_example()
    lengths = {fmt: units_per_pair(fmt) for fmt in ("continuous", "keypress", "word")}

    fig = plt.figure(figsize=(16, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[2.6, 1.0, 1.1])
    timeline = fig.add_subplot(grid[0, :])
    first_key, last_key = min(logged) - 0.5, max(logged) + 1.2
    t = np.arange(trace.size) / SFREQ
    view = (t >= first_key) & (t <= last_key)
    trace_view = trace[view] - np.median(trace[view])
    trace_view = np.convolve(trace_view, np.ones(51) / 51, mode="same")  # display-only smoothing
    trace_view -= np.convolve(trace_view, np.ones(1025) / 1025, mode="same")
    scale = 0.45 / (np.percentile(np.abs(trace_view), 99) + 1e-9)
    timeline.plot(t[view], 5.0 + trace_view * scale, color="0.55", linewidth=0.5)
    timeline.text(first_key, 5.55, "Raw typing EEG (one channel)", fontsize=9, color="0.35")

    rows = {
        "Continuous (unchanged)": 4.0,
        f"Keypress BEFORE: first {OLD_LIMIT} keys": 3.0,
        f"Keypress NOW: up to {NEW_LIMIT} keys": 2.0,
        "Word-level NEW: one epoch per typed word": 1.0,
    }
    starts = np.arange(min(logged) - PRE / SFREQ, max(logged) + (WINDOW - PRE) / SFREQ, WINDOW / SFREQ)
    for i, left in enumerate(starts):
        timeline.add_patch(Rectangle((left, 3.7), WINDOW / SFREQ, 0.6, facecolor="tab:blue",
                                     alpha=0.25 if i % 2 else 0.4, edgecolor="tab:blue"))
    for row_y, limit, color in ((3.0, OLD_LIMIT, "tab:orange"), (2.0, NEW_LIMIT, "tab:green")):
        for index, (time, char) in enumerate(keys):
            kept = index < limit
            timeline.add_patch(Rectangle((time - PRE / SFREQ, row_y - 0.3), WINDOW / SFREQ, 0.6,
                                         facecolor=color if kept else "none", alpha=0.18 if kept else 1,
                                         edgecolor=color if kept else "red", linestyle="-" if kept else ":"))
            timeline.plot([time, time], [row_y - 0.3, row_y + 0.3], color=color if kept else "red", linewidth=0.8)
    for word_times in words:
        anchor = min(word_times)
        timeline.add_patch(Rectangle((anchor - PRE / SFREQ, 0.7), WINDOW / SFREQ, 0.6,
                                     facecolor="tab:purple", alpha=0.35, edgecolor="tab:purple"))
        timeline.add_patch(Rectangle((anchor, 0.62), max(word_times) - anchor, 0.08, facecolor="tab:purple", alpha=0.8))
    for time, char in keys:
        timeline.text(time, 0.25, char if char.strip() else "␣", ha="center", fontsize=8)
    timeline.set_yticks(list(rows.values()))
    timeline.set_yticklabels(list(rows))
    timeline.set_ylim(0, 5.8)
    timeline.set_xlim(first_key, last_key)
    timeline.set_xlabel("Seconds from start of typing phase")
    timeline.set_title(f'Real trial: {pid} typed "{text}"  ({len(keys)} surviving keys, {len(words)} words)\n'
                       "Boxes = EEG units fed to the model (each 1 s: -200 to +800 ms). Red dotted keypress "
                       "boxes = dropped by the old 50-key limit. Purple bar = time spent typing the word.", fontsize=11)
    timeline.grid(axis="x", alpha=0.2)

    keypress = fig.add_subplot(grid[1, :])
    bins = np.arange(0, lengths["keypress"].max() + 5, 2)
    keypress.hist(lengths["keypress"], bins=bins, color="tab:orange", alpha=0.75)
    for limit, color, label in ((OLD_LIMIT, "red", "before"), (NEW_LIMIT, "tab:green", "now")):
        cut = int((lengths["keypress"] > limit).sum())
        keypress.axvline(limit, color=color, linewidth=2, linestyle="--")
        keypress.text(limit + 1, keypress.get_ylim()[1] * 0.85,
                      f"{label}: limit {limit}\n{cut} of {lengths['keypress'].size} pairs truncated "
                      f"({100 * cut / lengths['keypress'].size:.0f}%)", color=color, fontsize=10)
    keypress.set_title("Keypress format: keys per typing→response pair (all 882 trials)")
    keypress.set_xlabel("Number of keypress epochs")
    keypress.set_ylabel("Pairs")

    for column, (fmt, color, title) in enumerate((
            ("continuous", "tab:blue", "Continuous: 1 s windows per pair"),
            ("word", "tab:purple", "Word-level (new): word epochs per pair"))):
        axis = fig.add_subplot(grid[2, column])
        axis.hist(lengths[fmt], bins=np.arange(0, lengths[fmt].max() + 2, 1 if fmt == "word" else 3),
                  color=color, alpha=0.75)
        axis.axvline(np.median(lengths[fmt]), color="black", linestyle=":", label=f"median {np.median(lengths[fmt]):.0f}")
        axis.set_title(title)
        axis.set_xlabel("EEG units per pair")
        axis.set_ylabel("Pairs")
        axis.legend()

    fig.suptitle("Typing EEG formats: before vs typing v2", fontsize=15)
    output = Path(__file__).resolve().parent / "typing_formats_before_after.png"
    fig.savefig(output, dpi=160)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
