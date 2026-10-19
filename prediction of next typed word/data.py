"""Causal EEG-only features and next-word labels from typing exports."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np


@dataclass
class EEGSequence:
    participant: str
    sequence_id: str
    features: np.ndarray
    targets: list[str]
    examples: list[dict]


def sample_number(value) -> int | None:
    try:
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else None
    except (TypeError, ValueError):
        return None


def window_features(samples: np.ndarray, bins: int) -> np.ndarray:
    if samples.shape[1] < bins:
        raise ValueError("EEG window has fewer samples than feature bins")
    microvolts = samples * 1e6
    chunks = np.array_split(microvolts, bins, axis=1)
    means = np.stack([chunk.mean(axis=1) for chunk in chunks], axis=1)
    powers = np.stack([np.log1p(chunk.var(axis=1)) for chunk in chunks], axis=1)
    return np.concatenate([means.ravel(), powers.ravel()]).astype(np.float32)


def load_sequences(roots: list[Path], window_s: float = 1.0, bins: int = 16,
                   boundary: str = "completion") -> tuple[list[EEGSequence], dict]:
    if not math.isfinite(window_s) or window_s <= 0 or bins < 1:
        raise ValueError("window_s and bins must be positive")
    if boundary not in {"completion", "last_key"}:
        raise ValueError("boundary must be completion or last_key")
    groups = defaultdict(list)
    seen_roots = set()
    for root in roots:
        root = root.expanduser().resolve()
        if root in seen_roots:
            raise ValueError(f"Repeated dataset root: {root}")
        seen_roots.add(root)
        with (root / "typing_word_timings.csv").open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"participant_id", "trial", "word_index", "word", "typing_eeg_file",
                        "timing_status", "eeg_alignment_status", "identity_status",
                        "first_key_segment_sample", f"{boundary}_segment_sample"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing word-timing columns: {sorted(required - set(reader.fieldnames or []))}")
            for row in reader:
                if not row["typing_eeg_file"]:
                    continue
                path = (root / row["typing_eeg_file"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("EEG paths must remain inside their dataset root")
                groups[(row["participant_id"], str(path), row["trial"])].append(row)
    sequences = []
    skipped = Counter()
    channel_names = None
    for (participant, filename, trial), rows in sorted(groups.items()):
        indices = [sample_number(row["word_index"]) for row in rows]
        if any(index is None for index in indices) or len(set(indices)) != len(indices):
            raise ValueError(f"Invalid/duplicate word indices: {participant}, {filename}, trial {trial}")
        rows.sort(key=lambda row: int(float(row["word_index"])))
        raw = mne.io.read_raw_fif(filename, preload=False, verbose="ERROR")
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        names = [raw.ch_names[index] for index in picks]
        if not names:
            raise ValueError(f"No EEG-typed channels in {filename}")
        if channel_names is None:
            channel_names = names
        if names != channel_names:
            raise ValueError("EEG channel names/order differ across segments; harmonize exports first")
        sfreq = float(raw.info["sfreq"])
        window_samples = int(round(window_s * sfreq))
        if window_samples < bins:
            raise ValueError("Increase window_s or reduce bins for this sampling rate")
        features, targets, examples = [], [], []
        previous_cutoff = -1
        for current, target in zip(rows, rows[1:]):
            if any(row.get("timing_status") != "csv_validated" or
                   row.get("eeg_alignment_status") != "aligned" or
                   row.get("identity_status") not in {"consistent", "confirmed_by_manifest"}
                   for row in (current, target)):
                skipped["unvalidated_or_unverified_identity"] += 1
                continue
            if int(float(target["word_index"])) != int(float(current["word_index"])) + 1:
                skipped["nonadjacent_words"] += 1
                continue
            if boundary == "completion" and current.get("completion_kind") != "space":
                skipped["unreliable_completion"] += 1
                continue
            cutoff = sample_number(current.get(f"{boundary}_segment_sample"))
            target_first = sample_number(target.get("first_key_segment_sample"))
            if cutoff is None or target_first is None:
                skipped["missing_sample"] += 1
                continue
            if cutoff >= target_first or cutoff <= previous_cutoff:
                skipped["noncausal_word_order"] += 1
                continue
            start = cutoff - window_samples + 1
            if start < 0 or cutoff >= raw.n_times:
                skipped["window_outside_segment"] += 1
                continue
            samples = raw.get_data(picks=picks, start=start, stop=cutoff + 1)
            if not np.isfinite(samples).all():
                skipped["nonfinite_eeg"] += 1
                continue
            word = target["word"].casefold()
            if not word.strip():
                skipped["empty_target"] += 1
                continue
            features.append(window_features(samples, bins))
            targets.append(word)
            examples.append({"participant_id": participant, "trial": trial,
                             "question_id": current.get("question_id"),
                             "word_index": int(float(current["word_index"])),
                             "current_word_for_audit_only": current["word"], "target_word": word,
                             "eeg_file": filename, "sampling_rate_hz": sfreq,
                             "window_start_segment_sample": start,
                             "cutoff_segment_sample": cutoff,
                             "target_first_key_segment_sample": target_first,
                             "prediction_boundary": boundary})
            previous_cutoff = cutoff
        raw.close()
        if features:
            sequences.append(EEGSequence(participant, f"{participant}:{filename}:{trial}",
                                         np.stack(features), targets, examples))
    if not sequences:
        raise ValueError(f"No causal, validated EEG/next-word pairs found. Skipped: {dict(skipped)}")
    summary = {"participants": sorted({sequence.participant for sequence in sequences}),
               "sequences": len(sequences), "examples": sum(len(sequence.targets) for sequence in sequences),
               "channel_names": channel_names, "window_s": window_s, "bins": bins,
               "boundary": boundary, "feature_dim": sequences[0].features.shape[1], "skipped": dict(skipped)}
    return sequences, summary