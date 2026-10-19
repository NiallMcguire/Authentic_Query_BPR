"""Batch export Task 2 EEG phases using the local BMI4IRDS pipeline."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


csv.field_size_limit(sys.maxsize)
PARTICIPANT_PATTERN = re.compile(r"(?:sub-)?(P\d{3})", re.IGNORECASE)


def csv_identity(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"trigger_code", "trigger_time"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV lacks trigger_code/trigger_time columns")
        identities = set()
        dates = set()
        for row in reader:
            if row.get("participant"):
                identities.add(row["participant"].strip())
            for column in ("date", "expStart"):
                if row.get(column):
                    dates.add(row[column].strip())
    return {"internal_participants": sorted(identities), "internal_dates": sorted(dates)}


def discover_participants(base_dir: Path, selected: list[str] | None = None,
                          overrides: dict | None = None) -> list[dict]:
    records = []
    for folder in sorted(base_dir.iterdir()):
        match = PARTICIPANT_PATTERN.fullmatch(folder.name)
        if not folder.is_dir() or not match:
            continue
        participant = match.group(1).upper()
        if selected and participant not in selected:
            continue
        record = {"participant_id": participant, "source_folder": str(folder), "issues": []}
        override = (overrides or {}).get(participant, {})
        participant_paths = [folder, *(path for path in folder.rglob("*") if path.is_dir())]
        eeg_dirs = [path for path in participant_paths if path.name.lower() == "eeg"]
        psychopy_dirs = [path for path in participant_paths
                 if path.name.lower() in {"psychopy", "psyhcopy", "pyschopy"}]
        eeg_files = sorted(path for directory in eeg_dirs for path in directory.rglob("*.xdf"))
        if not eeg_files:
            eeg_files = sorted(path for directory in eeg_dirs for path in directory.rglob("*.vhdr"))
        csv_files = sorted(path for directory in psychopy_dirs for path in directory.rglob("*.csv"))
        for key, candidates in (("eeg_file", eeg_files), ("psychopy_csv", csv_files)):
            if key in override:
                chosen = Path(override[key]).expanduser()
                if not chosen.is_absolute():
                    chosen = base_dir / chosen
                candidates[:] = [chosen.resolve()] if chosen.is_file() else []
        valid_csvs = []
        for candidate in csv_files:
            try:
                identity = csv_identity(candidate)
            except (OSError, ValueError, csv.Error) as error:
                record.setdefault("rejected_csvs", []).append({"path": str(candidate), "reason": str(error)})
            else:
                valid_csvs.append((candidate, identity))
        record["eeg_candidates"] = [str(path) for path in eeg_files]
        record["psychopy_candidates"] = [str(path) for path, identity in valid_csvs]
        if len(eeg_files) != 1:
            record["issues"].append(f"Expected one EEG recording, found {len(eeg_files)}")
        else:
            record["eeg_file"] = str(eeg_files[0])
        if len(valid_csvs) != 1:
            record["issues"].append(f"Expected one PsychoPy event CSV, found {len(valid_csvs)}")
        else:
            path, identity = valid_csvs[0]
            record["psychopy_csv"] = str(path)
            record.update(identity)
            internal = identity["internal_participants"]
            record["identity_status"] = (
                "consistent" if internal == [participant] else "mismatch" if internal else "missing"
            )
            if override.get("confirmed_participant_id") == participant:
                record["identity_status"] = "confirmed_by_manifest"
                record["identity_confirmation"] = override.get("identity_confirmation", "User-supplied manifest")
            log_candidates = sorted(path.parent.glob("*.log"))
            record["psychopy_log_candidates"] = [str(candidate) for candidate in log_candidates]
            if "psychopy_log" in override:
                log = Path(override["psychopy_log"]).expanduser()
                if not log.is_absolute():
                    log = base_dir / log
                record["typing_log_status"] = "selected" if log.is_file() else "missing_override"
            else:
                log = path.with_suffix(".log")
                record["typing_log_status"] = "selected" if log.is_file() else (
                    "ambiguous" if log_candidates else "missing")
            if record["typing_log_status"] == "selected":
                record["psychopy_log"] = str(log.resolve())
        records.append(record)
    counts = {}
    for record in records:
        participant = record["participant_id"]
        counts[participant] = counts.get(participant, 0) + 1
    for record in records:
        if counts[record["participant_id"]] > 1:
            record["issues"].append("Duplicate normalized participant folder")
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", type=Path, default=Path.home() / "BMI4IRDS")
    parser.add_argument("--base-dir", type=Path, default=Path.home() / "Task 2 - Niall")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "extracted_dataset")
    parser.add_argument("--participants", nargs="+", help="Optional folder IDs, e.g. P007 P031")
    parser.add_argument("--input-manifest", type=Path,
                        help="JSON object keyed by participant ID with explicit eeg_file/psychopy_csv, optional psychopy_log, and identity confirmation")
    parser.add_argument("--expected-participants", type=int, default=30)
    parser.add_argument("--allow-unverified-identities", action="store_true",
                        help="Export conflicting/missing identities under folder IDs, flagged as unverified")
    parser.add_argument("--dry-run", action="store_true", help="Inventory inputs without loading EEG or writing output")
    return parser


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def index_results(results: dict, record: dict, output_dir: Path) -> tuple[list, list]:
    from SyncScripts.PhaseSegmentation.clock_alignment import make_time_mapper

    participant = record["participant_id"]
    raw = results["annotated_raw"]
    sfreq = float(raw.info["sfreq"])
    common = {"participant_id": participant, "identity_status": record["identity_status"],
              "source_eeg": record["eeg_file"], "source_psychopy": record["psychopy_csv"],
              "sampling_rate_hz": sfreq, "channel_names": json.dumps(raw.ch_names),
              "alignment_report": f"participants/{participant}/{participant}_alignment_report.json"}
    rows = []
    exclusions = []
    for phase, key, filename in (
        ("reading", "reading_epochs", f"{participant}_reading-epo.fif"),
        ("imagining_window", "imagining_epochs", f"{participant}_imagining_windows-epo.fif"),
        ("response_fixation", "response_epochs", f"{participant}_response_fixations-epo.fif"),
    ):
        epochs = results[key]
        if epochs is None:
            continue
        metadata = epochs.metadata.to_dict("records")
        for epoch_index, (item, event) in enumerate(zip(metadata, epochs.events)):
            start = int(event[0]) + int(round(float(epochs.times[0]) * sfreq)) - raw.first_samp
            end = start + len(epochs.times) - 1
            rows.append({**item, **common, "phase": phase, "epoch_index": epoch_index,
                         "eeg_file": f"participants/{participant}/{filename}",
                         "eeg_start_sample": start, "eeg_end_sample": end,
                         "n_samples": len(epochs.times), "sample_end_inclusive": True,
                         "eeg_start_time_s": start / sfreq + raw.first_time,
                         "eeg_end_time_s": end / sfreq + raw.first_time})
    for phase, key, folder in (
        ("imagining", "imagining_meta", "imagining_raw_segments"),
        ("typing", "typing_meta", "typing_raw_segments"),
        ("judgement", "judgement_stages", "judgement_raw_segments"),
    ):
        for item in results[key].to_dict("records"):
            if item.get("exclusion_reason"):
                exclusions.append({**item, **common, "phase": phase})
                continue
            trial = int(item["trial"])
            filename = (f"{participant}_trial-{trial:02d}_{item['stage']}_raw.fif" if phase == "judgement"
                        else f"{participant}_trial-{trial:02d}.fif")
            relative = f"participants/{participant}/{folder}/{filename}"
            if not (output_dir / relative).is_file():
                raise RuntimeError(f"Pipeline metadata points to missing EEG: {relative}")
            start = int(item["eeg_start_sample"])
            end = int(item["eeg_end_sample"])
            rows.append({**item, **common, "phase": phase, "eeg_file": relative,
                         "epoch_index": None, "n_samples": end - start + 1,
                         "sample_end_inclusive": True, "eeg_start_time_s": start / sfreq + raw.first_time,
                         "eeg_end_time_s": end / sfreq + raw.first_time})
    mapper = make_time_mapper(results["alignment_report"])
    for item in results["phase_timings"].to_dict("records"):
        for phase, onset_key, offset_key in (
            ("reading_routine", "read_imagine_onset", "read_imagine_offset"),
            ("imagining", "fixation_onset", "fixation_offset"),
            ("typing", "typing_onset", "typing_offset"),
        ):
            onset, offset = item.get(onset_key), item.get(offset_key)
            reason = ""
            if onset is None or offset is None or not (onset == onset and offset == offset):
                reason = "missing_phase_boundary"
            elif offset <= onset:
                reason = "nonpositive_phase_duration"
            elif phase == "typing" and item.get("excluded"):
                reason = str(item.get("exclusion_reason") or "excluded_by_phase_timings")
            else:
                mapped = mapper([onset, offset])
                if mapped[0] < raw.first_time or mapped[1] > raw.first_time + raw.times[-1]:
                    reason = "outside_available_eeg"
            if reason:
                exclusions.append({**item, **common, "phase": phase, "exclusion_reason": reason})
    return rows, exclusions


def fixation_exclusions(results: dict, config, record: dict) -> list:
    from SyncScripts.PhaseSegmentation import imagining_segments, io_utils, judgement_segments, reading_segments

    source = io_utils.load_psychopy_csv(config.psychopy_csv)
    candidates = reading_segments.collect_all_fixations(source, results["phase_timings"],
                                                       config.part_specs, config.reading)
    response = judgement_segments.detect_response_fixations(source, results["judgement_stages"], config.reading)
    exclusions = []
    for phase, table, key, onset_column in (
        ("reading", candidates, "reading_epochs", "onset"),
        ("response_fixation", response, "response_epochs", "fixation_onset"),
    ):
        epochs = results[key]
        retained = set() if epochs is None else {
            (item.get("trial"), item.get("stage"), item[onset_column])
            for item in epochs.metadata.to_dict("records")
        }
        for item in table.to_dict("records"):
            if (item.get("trial"), item.get("stage"), item[onset_column]) not in retained:
                exclusions.append({**item, "participant_id": record["participant_id"], "phase": phase,
                                   "identity_status": record["identity_status"],
                                   "exclusion_reason": "epoch_not_retained_by_library"})
    epochs = results["imagining_epochs"]
    retained_windows = set() if epochs is None else {
        (item["trial"], item["window_index"]) for item in epochs.metadata.to_dict("records")
    }
    for item in imagining_segments.build_imagining_metadata(results["phase_timings"]).to_dict("records"):
        for window_index in range(max(math.floor(item["duration_s"] - 1.0) + 1, 0)):
            if (item["trial"], window_index) not in retained_windows:
                onset = item["onset"] + window_index
                exclusions.append({**item, "participant_id": record["participant_id"],
                                   "identity_status": record["identity_status"], "phase": "imagining_window",
                                   "window_index": window_index, "window_onset": onset,
                                   "window_offset": onset + 1.0,
                                   "exclusion_reason": "window_not_retained_by_library"})
    return exclusions


def extract_batch(args, inventory: dict) -> int:
    library_dir = args.library_dir.expanduser().resolve()
    if not (library_dir / "SyncScripts" / "PhaseSegmentation" / "pipeline.py").is_file():
        raise ValueError(f"BMI4IRDS phase pipeline not found under {library_dir}")
    sys.path.insert(0, str(library_dir))
    import pandas as pd
    from SyncScripts.PhaseSegmentation.config import ImaginingParams, ParticipantConfig
    from SyncScripts.PhaseSegmentation.clock_alignment import make_time_mapper
    from SyncScripts.PhaseSegmentation.pipeline import run_pipeline

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty; choose a new --output-dir: {output}")
    if output == args.base_dir or args.base_dir in output.parents or output in args.base_dir.parents:
        raise ValueError("Output directory must be separate from source dataset")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {**inventory, "created_utc": datetime.now(timezone.utc).isoformat(),
                "library_dir": str(library_dir), "python_executable": sys.executable,
                "library_source_sha256": {
                    str(path.relative_to(library_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted((library_dir / "SyncScripts" / "PhaseSegmentation").glob("*.py"))
                }, "typing_annotation_source_sha256": hashlib.sha256(
                    Path(__file__).with_name("typing_annotations.py").read_bytes()).hexdigest(),
                "package_versions": {name: importlib.metadata.version(name)
                                          for name in ("numpy", "pandas", "mne", "pyxdf")},
                "parameters": {"imagining_window_s": 1.0, "min_fixation_ms": 100.0,
                               "max_gap_frames": 1, "epoch_tmin": -0.2, "epoch_tmax": 0.8,
                               "baseline": [-0.2, 0.0], "typing_log_csv_tolerance_s": 0.05,
                               "typing_eeg_alignment_requires": "markers and passes_frame_threshold"},
                "participant_results": []}
    all_rows, all_exclusions, all_timings, all_gaze = [], [], [], []
    all_keys, all_typing_trials, all_words = [], [], []
    write_json(output / "dataset_manifest.json", manifest)
    for record in inventory["participants"]:
        participant = record["participant_id"]
        status = {**record, "status": "blocked"}
        reasons = list(record["issues"])
        if record.get("identity_status") not in {"consistent", "confirmed_by_manifest"}:
            if not args.allow_unverified_identities:
                reasons.append("Identity is unverified; confirm via input manifest before exporting")
            else:
                status["identity_unverified"] = True
        status["blocking_reasons"] = reasons
        if not reasons:
            participant_output = output / "participants" / participant
            participant_output.mkdir(parents=True)
            log_path = participant_output / "extraction.log"
            status["log_file"] = str(log_path.relative_to(output))
            try:
                with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    config = ParticipantConfig(participant_id=participant, psychopy_csv=record["psychopy_csv"],
                                               eeg_file=record["eeg_file"], output_dir=participant_output,
                                               imagining=ImaginingParams(window_lengths_s=(1.0,)))
                    results = run_pipeline(config)
                    rows, exclusions = index_results(results, record, output)
                    exclusions.extend(fixation_exclusions(results, config, record))
                    from typing_annotations import extract_annotations

                    phases = results["phase_timings"].to_dict("records")
                    try:
                        keys, typing_trials, words = extract_annotations(
                            record, phases, rows, results["annotated_raw"], results["alignment_report"])
                    except Exception as error:
                        traceback.print_exc()
                        status["typing_annotation_error"] = f"{type(error).__name__}: {error}"
                        keys, typing_trials, words = extract_annotations(
                            {key: value for key, value in record.items() if key != "psychopy_log"},
                            phases, rows)
                        for item in typing_trials:
                            item["timing_status"] = "annotation_error"
                    trial_annotations = {int(item["trial"]): item for item in typing_trials
                                         if item.get("trial") is not None}
                    for row in rows:
                        if row["phase"] == "typing":
                            annotation = trial_annotations.get(int(row["trial"]), {})
                            row["typing_timing_status"] = annotation.get("timing_status", "unavailable")
                            row["typing_eeg_alignment_status"] = annotation.get("eeg_alignment_status", "unavailable")
                    status["typing_annotation_counts"] = {"keystrokes": len(keys), "words": len(words),
                        "validated_trials": sum(item["timing_status"] == "csv_validated" for item in typing_trials),
                        "aligned_trials": sum(item["eeg_alignment_status"] == "aligned" for item in typing_trials)}
                    mapper = make_time_mapper(results["alignment_report"])
                    raw = results["annotated_raw"]
                    gaze_rows = []
                    for item in results["judgement_gaze_timeline"].to_dict("records"):
                        onset, offset = mapper([item["onset"], item["offset"]])
                        gaze_rows.append({**item, "participant_id": participant,
                                          "identity_status": record["identity_status"],
                                          "eeg_onset_s": float(onset), "eeg_offset_s": float(offset),
                                          "eeg_start_sample": int(round((onset - raw.first_time) * raw.info["sfreq"])),
                                          "eeg_end_sample": int(round((offset - raw.first_time) * raw.info["sfreq"])),
                                          "outside_available_eeg": bool(onset < raw.first_time or
                                              offset > raw.first_time + raw.times[-1])})
                    counts = {}
                    for row in rows:
                        counts[row["phase"]] = counts.get(row["phase"], 0) + 1
                    status.update(status="completed", segment_counts=counts,
                                  alignment=results["alignment_report"].to_dict(),
                                  sanity_results=results["sanity_results"])
                    all_rows.extend(rows)
                    all_exclusions.extend(exclusions)
                    all_timings.extend({**item, "participant_id": participant,
                                        "identity_status": record["identity_status"]}
                                       for item in results["phase_timings"].to_dict("records"))
                    all_gaze.extend(gaze_rows)
                    all_keys.extend(keys)
                    all_typing_trials.extend(typing_trials)
                    all_words.extend(words)
                    del results
                    del raw
            except Exception as error:
                with log_path.open("a", encoding="utf-8") as log:
                    traceback.print_exc(file=log)
                status.update(status="failed", error=f"{type(error).__name__}: {error}")
        manifest["participant_results"].append(status)
        write_json(output / "dataset_manifest.json", manifest)
        print(f"{participant}: {status['status']}", flush=True)
    for filename, rows, columns in (
        ("segments.csv", all_rows, ["participant_id", "phase", "eeg_file", "epoch_index"]),
        ("exclusions.csv", all_exclusions, ["participant_id", "phase", "exclusion_reason"]),
        ("phase_timings.csv", all_timings, ["participant_id", "trial"]),
        ("judgement_gaze_timeline.csv", all_gaze, ["participant_id", "trial"]),
        ("typing_keystrokes.csv", all_keys, ["participant_id", "part", "trial", "log_time_s", "key",
                            "timing_status", "keypress_eeg_sample", "keypress_segment_sample"]),
        ("typing_trials.csv", all_typing_trials, ["participant_id", "part", "trial", "timing_status",
                            "eeg_alignment_status", "typing_eeg_file"]),
        ("typing_word_timings.csv", all_words, ["participant_id", "part", "trial", "word_index", "word",
                               "completion_s", "completion_kind", "timing_status",
                               "completion_eeg_sample", "completion_segment_sample"]),
    ):
        (pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)).to_csv(output / filename, index=False)
    completed = sum(item["status"] == "completed" for item in manifest["participant_results"])
    manifest["completed_participants"] = completed
    manifest["total_segments"] = len(all_rows)
    manifest["complete_requested_cohort"] = bool(completed) and completed == len(inventory["participants"]) and not inventory.get("requested_but_missing")
    if not args.participants:
        manifest["complete_requested_cohort"] &= completed == args.expected_participants
    write_json(output / "dataset_manifest.json", manifest)
    print(f"Exported {len(all_rows)} segments from {completed}/{len(inventory['participants'])} participants to {output}")
    return 0 if manifest["complete_requested_cohort"] else 1


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.expected_participants < 1:
        parser.error("--expected-participants must be positive")
    args.base_dir = args.base_dir.expanduser().resolve()
    if not args.base_dir.is_dir():
        parser.error(f"Dataset directory does not exist: {args.base_dir}")
    selected = [participant.upper() for participant in args.participants] if args.participants else None
    overrides = {}
    if args.input_manifest:
        with args.input_manifest.expanduser().open(encoding="utf-8") as stream:
            overrides = json.load(stream)
        if not isinstance(overrides, dict) or any(not isinstance(value, dict) for value in overrides.values()):
            parser.error("Input manifest must be an object keyed by participant ID, with object values")
    records = discover_participants(args.base_dir, selected, overrides)
    inventory = {"expected_participants": args.expected_participants, "discovered_participants": len(records),
                 "participants": records}
    if selected:
        inventory["requested_but_missing"] = sorted(set(selected) - {row["participant_id"] for row in records})
    if args.dry_run:
        print(json.dumps(inventory, indent=2))
        return int(not records or (not selected and len(records) != args.expected_participants)
                   or bool(inventory.get("requested_but_missing")) or any(
            row["issues"] or (row.get("identity_status") not in {"consistent", "confirmed_by_manifest"}
                              and not args.allow_unverified_identities)
            for row in records
        ))
    try:
        return extract_batch(args, inventory)
    except (ImportError, ValueError, OSError) as error:
        print(f"Extraction setup failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())