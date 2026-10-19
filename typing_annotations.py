"""Recover typing annotations without replacing continuous typing EEG."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path


EVENT = re.compile(r"^\s*([\d.]+)\s+(DATA|EXP)\s+(.*)$")
WINDOW = re.compile(r"textbox(_[23])?: autoDraw = (True|False)$")
CHARACTERS = {
    "space": " ", "comma": ",", "period": ".", "slash": "/",
    "backslash": "\\", "semicolon": ";", "apostrophe": "'", "minus": "-",
    "equal": "=", "bracketleft": "[", "bracketright": "]", "grave": "`",
}
SHIFTED = dict(zip("1234567890-=[]\\;',./`", "!@#$%^&*()_+{}|:\"<>?~"))
MODIFIERS = {"lshift", "rshift", "capslock"}


def read_windows(path: Path) -> list[dict]:
    windows = []
    active = None
    with path.open(encoding="utf-8-sig", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            event = EVENT.match(line.rstrip())
            if not event:
                continue
            timestamp, level, message = event.groups()
            timestamp = float(timestamp)
            boundary = WINDOW.fullmatch(message)
            if boundary:
                suffix, enabled = boundary.groups()
                part = int(suffix[1:]) if suffix else 1
                if enabled == "True":
                    if active is not None:
                        windows.append(active)
                    active = {"part": part, "log_start_s": timestamp,
                              "log_end_s": None, "keys": []}
                elif active is not None and active["part"] == part:
                    active["log_end_s"] = timestamp
                    windows.append(active)
                    active = None
            elif level == "DATA" and message.startswith("Keypress: ") and active is not None:
                active["keys"].append({"log_time_s": timestamp,
                                       "key": message.removeprefix("Keypress: "),
                                       "log_line": line_number})
    if active is not None:
        windows.append(active)
    return windows


def replay(keys: list[dict]) -> tuple[list, list, list]:
    characters, unsupported, submissions = [], [], []
    cursor = 0
    for event in keys:
        key, timestamp = event["key"], event["log_time_s"]
        if key in MODIFIERS:
            continue
        if key in {"return", "enter", "num_enter"}:
            submissions.append(timestamp)
        elif key == "backspace":
            if cursor:
                cursor -= 1
                characters.pop(cursor)
        elif key == "delete":
            if cursor < len(characters):
                characters.pop(cursor)
        elif key in {"left", "right"}:
            cursor = max(0, min(len(characters), cursor + (1 if key == "right" else -1)))
        elif key in {"home", "end"}:
            cursor = 0 if key == "home" else len(characters)
        else:
            character = key if len(key) == 1 else CHARACTERS.get(key)
            if character is None:
                unsupported.append(key)
            else:
                characters.insert(cursor, (character, timestamp))
                cursor += 1
    return characters, sorted(set(unsupported)), submissions


def text_matches(characters: list, submitted: str) -> bool:
    if len(characters) != len(submitted):
        return False
    return all(actual.casefold() == expected.casefold() or SHIFTED.get(actual) == expected
               for (actual, timestamp), expected in zip(characters, submitted))


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def extract_annotations(record: dict, phases: list[dict], segments: list[dict],
                        raw=None, report=None, tolerance_s: float = 0.05) -> tuple[list, list, list]:
    windows = read_windows(Path(record["psychopy_log"])) if record.get("psychopy_log") else []
    starts = {part: [] for part in (1, 2, 3)}
    if windows:
        with Path(record["psychopy_csv"]).open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                for part in starts:
                    component = "textbox" if part == 1 else f"textbox_{part}"
                    value = row.get(f"{component}.started")
                    if finite(value):
                        starts[part].append(float(value))
    used = set()
    trials, keystrokes, words = [], [], []
    segment_by_trial = {int(item["trial"]): item for item in segments if item["phase"] == "typing"}
    alignment_ok = bool(report is not None and report.success and report.passes_frame_threshold
                        and report.method == "markers")

    def mapped_fields(timestamp, prefix, segment, reliable):
        fields = {f"{prefix}_eeg_time_s": None, f"{prefix}_eeg_sample": None,
                  f"{prefix}_segment_sample": None, f"{prefix}_outside_available_eeg": None}
        if not reliable or not finite(timestamp) or raw is None:
            return fields
        eeg_time = report.a * float(timestamp) + report.b
        sample = int(round((eeg_time - raw.first_time) * raw.info["sfreq"]))
        outside = not (raw.first_time <= eeg_time <= raw.first_time + raw.times[-1])
        fields.update({f"{prefix}_eeg_time_s": eeg_time,
                       f"{prefix}_outside_available_eeg": outside})
        if not outside:
            fields[f"{prefix}_eeg_sample"] = sample
            if segment and segment["eeg_start_sample"] <= sample <= segment["eeg_end_sample"]:
                fields[f"{prefix}_segment_sample"] = sample - segment["eeg_start_sample"]
        return fields

    for phase in phases:
        trial = int(phase["trial"])
        part = int(phase["part"])
        segment = segment_by_trial.get(trial)
        common = {key: phase.get(key) for key in (
            "part", "trial", "trial_in_part", "question_id", "category", "condition", "chatbot_condition")}
        common.update(participant_id=record["participant_id"], identity_status=record["identity_status"],
                      source_psychopy=record["psychopy_csv"], source_log=record.get("psychopy_log"),
                      typing_eeg_file=segment["eeg_file"] if segment else None,
                      alignment_report=f"participants/{record['participant_id']}/{record['participant_id']}_alignment_report.json")
        status = ("unmatched_log_window" if windows else
              "no_typing_windows" if record.get("psychopy_log") else
              record.get("typing_log_status", "missing"))
        window = None
        if windows and finite(phase.get("typing_onset")) and finite(phase.get("typing_offset")):
            candidates = [(index, item) for index, item in enumerate(windows)
                          if index not in used and item["part"] == part
                          and phase["typing_onset"] - tolerance_s <= item["log_start_s"]
                          <= phase["typing_offset"]]
            if len(candidates) == 1:
                index, window = candidates[0]
                used.add(index)
            status = "unmatched_log_window" if not candidates else "ambiguous_log_window"
        if window is None:
            trials.append({**common, "timing_status": status, "eeg_alignment_status": "unavailable"})
            continue
        characters, unsupported, submissions = replay(window["keys"])
        submitted = phase.get("typed_text")
        submitted = submitted.rstrip("\r\n") if isinstance(submitted, str) else None
        clock_ok = (sum(abs(value - window["log_start_s"]) <= tolerance_s for value in starts[part]) == 1
                    and finite(window["log_end_s"])
                    and abs(window["log_end_s"] - phase["typing_offset"]) <= tolerance_s)
        status = ("incomplete_log_window" if window["log_end_s"] is None else
                  "unsupported_keys" if unsupported else
                  "missing_submitted_text" if submitted is None else
                  "text_mismatch" if not text_matches(characters, submitted) else
                  "clock_mismatch" if not clock_ok else "csv_validated")
        reliable = status == "csv_validated" and alignment_ok
        common.update(timing_status=status, eeg_alignment_status=("aligned" if reliable else "unavailable"))
        trials.append({**common, "typing_onset": phase.get("typing_onset"),
                       "typing_offset": phase.get("typing_offset"),
                       "log_start_s": window["log_start_s"], "log_end_s": window["log_end_s"],
                       "log_end_minus_csv_offset_s": (window["log_end_s"] - phase["typing_offset"]
                                                       if finite(window["log_end_s"]) else None),
                       "log_csv_clock_verified": clock_ok,
                       "reconstructed_text": "".join(character for character, timestamp in characters),
                       "submitted_text": submitted, "unsupported_keys": ";".join(unsupported)})
        for key_index, event in enumerate(window["keys"], 1):
            keystrokes.append({**common, **event, "key_index": key_index,
                               **mapped_fields(event["log_time_s"], "keypress", segment, reliable)})
        if status != "csv_validated":
            continue
        for word_index, match in enumerate(re.finditer(r"\S+", submitted), 1):
            timestamps = [timestamp for character, timestamp in characters[match.start():match.end()]]
            last = max(timestamps)
            if match.end() < len(characters):
                completion = characters[match.end()][1]
                kind = "space"
            else:
                completion = next((value for value in submissions if value >= last), window["log_end_s"])
                kind = "submission" if completion in submissions else "routine_end_proxy"
            if completion < last:
                completion, kind = None, "edited_after_boundary"
            word = {**common, "word_index": word_index, "word": match.group(),
                    "first_surviving_key_s": min(timestamps), "last_surviving_key_s": last,
                    "completion_s": completion, "completion_kind": kind}
            for prefix, timestamp in (("first_key", min(timestamps)), ("last_key", last),
                                      ("completion", completion)):
                word.update(mapped_fields(timestamp, prefix, segment, reliable))
            words.append(word)
    for index, window in enumerate(windows):
        if index in used:
            continue
        common = {"participant_id": record["participant_id"], "identity_status": record["identity_status"],
                  "source_log": record["psychopy_log"], "part": window["part"],
                  "trial": None, "timing_status": "unmatched_log_window", "eeg_alignment_status": "unavailable"}
        trials.append({**common, "log_start_s": window["log_start_s"], "log_end_s": window["log_end_s"]})
        keystrokes.extend({**common, **event, "key_index": key_index}
                          for key_index, event in enumerate(window["keys"], 1))
    return keystrokes, trials, words