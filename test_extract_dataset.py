"""Focused tests for safe participant discovery and source selection."""

import contextlib
import csv
import io
import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import extract_dataset
import typing_annotations


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)

    def participant(self, folder="P007", internal="P007", psychopy_folder="Psychopy"):
        root = self.base / folder
        eeg = root / "eeg" / "recording.xdf"
        eeg.parent.mkdir(parents=True)
        eeg.touch()
        source = root / psychopy_folder / "experiment.csv"
        source.parent.mkdir()
        with source.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["participant", "trigger_code", "trigger_time", "expStart"])
            writer.writerow([internal, 1, 10.0, "2026-02-01"])
        return eeg, source

    def test_case_and_spelling_variants(self):
        self.participant(folder="sub-P031", internal="P031", psychopy_folder="psyhcopy")
        row, = extract_dataset.discover_participants(self.base)
        self.assertEqual(row["participant_id"], "P031")
        self.assertEqual(row["identity_status"], "consistent")
        self.assertEqual(row["issues"], [])

    def test_nested_input_directories_and_alternate_spelling(self):
        eeg, _ = self.participant(folder="P026", internal="P026", psychopy_folder="Pyschopy")
        nested_eeg = self.base / "P026" / "ses-S003" / "EEG" / eeg.name
        nested_eeg.parent.mkdir(parents=True)
        eeg.rename(nested_eeg)
        row, = extract_dataset.discover_participants(self.base)
        self.assertEqual(row["eeg_file"], str(nested_eeg))
        self.assertEqual(row["identity_status"], "consistent")
        self.assertEqual(row["issues"], [])

    def test_identity_mismatch_is_not_corrected_automatically(self):
        self.participant(internal="P004")
        row, = extract_dataset.discover_participants(self.base)
        self.assertEqual(row["identity_status"], "mismatch")
        self.assertEqual(row["internal_participants"], ["P004"])

    def test_multiple_eeg_files_require_selection(self):
        eeg, source = self.participant()
        eeg.with_name("recording_old1.xdf").touch()
        row, = extract_dataset.discover_participants(self.base)
        self.assertNotIn("eeg_file", row)
        self.assertTrue(row["issues"])
        override = {"P007": {"eeg_file": str(eeg.relative_to(self.base))}}
        chosen, = extract_dataset.discover_participants(self.base, overrides=override)
        self.assertEqual(chosen["eeg_file"], str(eeg))
        self.assertEqual(chosen["issues"], [])

    def test_manifest_confirmation_preserves_original_identity(self):
        self.participant(internal="P004")
        override = {"P007": {"confirmed_participant_id": "P007", "identity_confirmation": "Verified recording sheet"}}
        row, = extract_dataset.discover_participants(self.base, overrides=override)
        self.assertEqual(row["identity_status"], "confirmed_by_manifest")
        self.assertEqual(row["internal_participants"], ["P004"])

    def test_missing_manifest_path_does_not_fall_back(self):
        self.participant()
        row, = extract_dataset.discover_participants(self.base, overrides={"P007": {"eeg_file": "missing.xdf"}})
        self.assertNotIn("eeg_file", row)
        self.assertTrue(row["issues"])

    def test_non_event_csv_is_rejected(self):
        eeg, source = self.participant()
        source.write_text("participant,value\nP007,1\n")
        row, = extract_dataset.discover_participants(self.base)
        self.assertNotIn("psychopy_csv", row)
        self.assertIn("trigger_code", row["rejected_csvs"][0]["reason"])

    def test_duplicate_normalized_ids_are_blocked(self):
        self.participant()
        self.participant(folder="sub-P007")
        rows = extract_dataset.discover_participants(self.base)
        self.assertTrue(all("Duplicate normalized participant folder" in row["issues"] for row in rows))

    def test_optional_log_selection(self):
        _, source = self.participant()
        row, = extract_dataset.discover_participants(self.base)
        self.assertEqual(row["typing_log_status"], "missing")
        self.assertEqual(row["issues"], [])
        source.with_name("other.log").touch()
        row, = extract_dataset.discover_participants(self.base)
        self.assertNotIn("psychopy_log", row)
        source.with_suffix(".log").touch()
        row, = extract_dataset.discover_participants(self.base)
        self.assertEqual(row["psychopy_log"], str(source.with_suffix(".log")))
        row, = extract_dataset.discover_participants(
            self.base, overrides={"P007": {"psychopy_log": "missing.log"}})
        self.assertEqual(row["typing_log_status"], "missing_override")
        self.assertNotIn("psychopy_log", row)
        self.assertEqual(row["issues"], [])

    def test_dry_run_reports_cohort_gap_without_writing(self):
        self.participant()
        output = self.base / "output"
        arguments = ["extract_dataset.py", "--base-dir", str(self.base), "--output-dir", str(output), "--dry-run"]
        capture = io.StringIO()
        with patch("sys.argv", arguments), contextlib.redirect_stdout(capture):
            status = extract_dataset.main()
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(capture.getvalue())["discovered_participants"], 1)
        self.assertFalse(output.exists())


class TypingAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "experiment.csv"
        source.write_text("textbox.started\n10.0\n")
        self.log = source.with_suffix(".log")
        self.record = dict(participant_id="P012", identity_status="consistent",
                           psychopy_csv=str(source), psychopy_log=str(self.log), typing_log_status="selected")
        self.phase = dict(part=1, trial=17, question_id="Q1", typing_onset=9.99,
                          typing_offset=15.0, typed_text="hi? ok\n")
        self.segment = dict(phase="typing", trial=17, eeg_file="typing.fif",
                            eeg_start_sample=999, eeg_end_sample=1500)
        self.raw = SimpleNamespace(first_time=5.0, times=[0.0, 20.0], info={"sfreq": 100.0})
        self.report = SimpleNamespace(success=True, passes_frame_threshold=True, method="markers", a=1.0, b=5.0)
        self.events = [(11, "h"), (11.1, "x"), (11.2, "backspace"), (11.3, "i"),
                       (11.4, "slash"), (12, "space"), (12.1, "o"), (12.2, "k"), (14.9, "return")]

    def run_extract(self, events=None, end=True, **kwargs):
        lines = ["10.0000 EXP textbox: autoDraw = True"]
        lines.extend(f"{timestamp:.4f} DATA Keypress: {key}" for timestamp, key in
                     (self.events if events is None else events))
        if end:
            lines.append("15.0000 EXP textbox: autoDraw = False")
        self.log.write_text("\n".join(lines))
        return typing_annotations.extract_annotations(self.record, [self.phase], [self.segment],
                                                       self.raw, self.report, **kwargs)

    def test_reconstruction_corrections_shifted_punctuation_and_mapping(self):
        keys, trials, words = self.run_extract()
        self.assertEqual(len(keys), 9)
        self.assertEqual(trials[0]["timing_status"], "csv_validated")
        self.assertEqual(trials[0]["trial"], 17)
        self.assertEqual([item["word"] for item in words], ["hi?", "ok"])
        self.assertEqual(words[0]["completion_s"], 12)
        self.assertEqual(words[0]["completion_eeg_time_s"], 17)
        self.assertEqual(words[0]["completion_eeg_sample"], 1200)
        self.assertEqual(words[0]["completion_segment_sample"], 201)
        self.assertEqual(words[1]["completion_kind"], "submission")

    def test_mismatch_and_unsupported_keys_preserve_raw_events_only(self):
        self.phase["typed_text"] = "something else"
        keys, trials, words = self.run_extract()
        self.assertEqual(trials[0]["timing_status"], "text_mismatch")
        self.assertTrue(keys)
        self.assertFalse(words)
        self.assertIsNone(keys[0]["keypress_eeg_sample"])
        keys, trials, words = self.run_extract(events=self.events + [(14.95, "lctrl")])
        self.assertEqual(trials[0]["timing_status"], "unsupported_keys")
        self.assertFalse(words)

    def test_clock_mismatch_does_not_get_eeg_samples(self):
        Path(self.record["psychopy_csv"]).write_text("textbox.started\n8.0\n")
        keys, trials, words = self.run_extract()
        self.assertEqual(trials[0]["timing_status"], "clock_mismatch")
        self.assertFalse(words)
        self.assertIsNone(keys[0]["keypress_eeg_sample"])

    def test_low_quality_alignment_keeps_log_words_without_eeg_indices(self):
        self.report.passes_frame_threshold = False
        keys, trials, words = self.run_extract()
        self.assertEqual(trials[0]["timing_status"], "csv_validated")
        self.assertEqual(trials[0]["eeg_alignment_status"], "unavailable")
        self.assertTrue(words)
        self.assertIsNone(words[0]["completion_eeg_sample"])

    def test_missing_log_and_incomplete_window(self):
        record = {key: value for key, value in self.record.items() if key != "psychopy_log"}
        record["typing_log_status"] = "missing"
        keys, trials, words = typing_annotations.extract_annotations(record, [self.phase], [self.segment])
        self.assertFalse(keys)
        self.assertFalse(words)
        self.assertEqual(trials[0]["typing_eeg_file"], "typing.fif")
        self.assertEqual(trials[0]["timing_status"], "missing")
        keys, trials, words = self.run_extract(end=False)
        self.assertEqual(trials[0]["timing_status"], "incomplete_log_window")
        self.assertTrue(keys)
        self.assertFalse(words)

    def test_late_correction_leaves_earlier_space_completion_blank(self):
        self.phase["typed_text"] = "a c"
        events = [(11, "a"), (12, "space"), (12.1, "c"), (13, "home"),
                  (13.1, "delete"), (14, "a")]
        keys, trials, words = self.run_extract(events=events)
        self.assertEqual(trials[0]["timing_status"], "csv_validated")
        self.assertIsNone(words[0]["completion_s"])
        self.assertEqual(words[0]["completion_kind"], "edited_after_boundary")
        self.assertEqual(words[1]["completion_kind"], "routine_end_proxy")

    def test_outside_recording_and_missing_segment(self):
        self.raw.times = [0.0, 6.5]
        keys, trials, words = self.run_extract()
        self.assertTrue(words[0]["completion_outside_available_eeg"])
        self.assertIsNone(words[0]["completion_eeg_sample"])
        self.assertIsNone(words[0]["completion_segment_sample"])
        self.segment["phase"] = "reading"
        keys, trials, words = self.run_extract()
        self.assertIsNone(trials[0]["typing_eeg_file"])

    def test_empty_log_and_missing_phase_boundaries_are_unavailable(self):
        self.log.write_text("1.0 DATA Keypress: space\n")
        keys, trials, words = typing_annotations.extract_annotations(self.record, [self.phase], [])
        self.assertEqual(trials[0]["timing_status"], "no_typing_windows")
        self.phase["typing_onset"] = None
        keys, trials, words = self.run_extract()
        self.assertEqual(trials[0]["timing_status"], "unmatched_log_window")
        self.assertTrue(keys)
        self.assertTrue(all(item["trial"] is None for item in keys))
        self.assertFalse(words)


class ExportIntegrityTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("BMI4IRDS_TEST_DATASET"), "Set BMI4IRDS_TEST_DATASET to an exported dataset")
    def test_export_matches_saved_eeg(self):
        import mne
        import pandas as pd

        root = Path(os.environ["BMI4IRDS_TEST_DATASET"])
        manifest = json.loads((root / "dataset_manifest.json").read_text())
        table = pd.read_csv(root / "segments.csv")
        completed = {item["participant_id"] for item in manifest["participant_results"]
                     if item["status"] == "completed"}
        self.assertEqual(len(table), manifest["total_segments"])
        self.assertEqual(set(table.participant_id), completed)
        self.assertTrue((table.eeg_end_sample - table.eeg_start_sample + 1 == table.n_samples).all())
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="This filename.*", category=RuntimeWarning)
            for filename, group in table[table.epoch_index.notna()].groupby("eeg_file"):
                epochs = mne.read_epochs(root / filename, preload=False, verbose=False)
                self.assertEqual(len(epochs), len(group))
                self.assertEqual(list(group.epoch_index.astype(int)), list(range(len(epochs))))
                self.assertTrue((group.n_samples == len(epochs.times)).all())
                self.assertTrue((group.sampling_rate_hz == epochs.info["sfreq"]).all())
            for item in table[table.epoch_index.isna()].to_dict("records"):
                raw = mne.io.read_raw_fif(root / item["eeg_file"], preload=False, verbose=False)
                self.assertEqual(raw.n_times, item["n_samples"])
                self.assertEqual(raw.info["sfreq"], item["sampling_rate_hz"])
                self.assertEqual(raw.ch_names, json.loads(item["channel_names"]))
        gaze = pd.read_csv(root / "judgement_gaze_timeline.csv")
        self.assertTrue({"eeg_onset_s", "eeg_offset_s", "eeg_start_sample", "eeg_end_sample",
                         "outside_available_eeg"}.issubset(gaze.columns))
        for participant, group in gaze.groupby("participant_id"):
            report = json.loads((root / "participants" / participant / f"{participant}_alignment_report.json").read_text())
            expected = group.onset * report["a"] + report["b"]
            self.assertTrue(((group.eeg_onset_s - expected).abs() < 1e-6).all())

    @unittest.skipUnless(os.environ.get("BMI4IRDS_TEST_DATASET"), "Set BMI4IRDS_TEST_DATASET to an exported dataset")
    def test_typing_annotations_match_saved_eeg(self):
        import mne
        import numpy as np
        import pandas as pd

        root = Path(os.environ["BMI4IRDS_TEST_DATASET"])
        segments = pd.read_csv(root / "segments.csv")
        keys = pd.read_csv(root / "typing_keystrokes.csv")
        trials = pd.read_csv(root / "typing_trials.csv")
        words = pd.read_csv(root / "typing_word_timings.csv")
        for participant, group in words.groupby("participant_id"):
            directory = root / "participants" / participant
            report = json.loads((directory / f"{participant}_alignment_report.json").read_text())
            raw = mne.io.read_raw_fif(directory / f"{participant}_annotated_raw.fif", preload=False, verbose=False)
            aligned = group[group.eeg_alignment_status.eq("aligned")]
            for source_column, prefix in (("first_surviving_key_s", "first_key"),
                                          ("last_surviving_key_s", "last_key"), ("completion_s", "completion")):
                subset = aligned[aligned[f"{prefix}_eeg_sample"].notna()]
                expected_times = subset[source_column] * report["a"] + report["b"]
                np.testing.assert_allclose(subset[f"{prefix}_eeg_time_s"], expected_times)
                expected_samples = np.rint((expected_times - raw.first_time) * raw.info["sfreq"])
                np.testing.assert_array_equal(subset[f"{prefix}_eeg_sample"], expected_samples)
            for item in segments[segments.participant_id.eq(participant) & segments.phase.eq("typing")].to_dict("records"):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="This filename.*", category=RuntimeWarning)
                    segment = mne.io.read_raw_fif(root / item["eeg_file"], preload=False, verbose=False)
                selected = aligned[aligned.trial.eq(item["trial"]) & aligned.completion_segment_sample.notna()]
                for word in selected.to_dict("records"):
                    sample = int(word["completion_eeg_sample"])
                    relative = int(word["completion_segment_sample"])
                    self.assertEqual(relative, sample - item["eeg_start_sample"])
                    self.assertGreaterEqual(relative, 0)
                    self.assertLess(relative, segment.n_times)
                    np.testing.assert_allclose(segment.get_data(start=relative, stop=relative + 1),
                                               raw.get_data(start=sample, stop=sample + 1))
        self.assertTrue(words.timing_status.eq("csv_validated").all())
        self.assertTrue(trials.loc[trials.typing_eeg_file.notna(), "trial"].notna().all())
        self.assertTrue(keys.loc[keys.eeg_alignment_status.ne("aligned"), "keypress_eeg_sample"].isna().all())


if __name__ == "__main__":
    unittest.main()