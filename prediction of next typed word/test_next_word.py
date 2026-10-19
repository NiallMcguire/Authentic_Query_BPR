"""Focused causal-data and next-word model tests."""

import csv
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import mne
import numpy as np
import torch

from data import EEGSequence, load_sequences
from model import ARCHITECTURES, EEGNextWord
from train import SequenceDataset, build_parser, collate_sequences, evaluate, fit_training_statistics, run, split_participants


class CausalDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.samples = np.stack([np.arange(400), np.arange(400) * 2]) * 1e-6
        self.rows = []
        for index, (word, first, completion) in enumerate(
                [("british", 20, 100), ("prime", 101, 200), ("minister", 201, 300)], 1):
            self.rows.append(dict(participant_id="P012", trial="3", word_index=index, word=word,
                                  typing_eeg_file="typing_raw.fif", timing_status="csv_validated",
                                  eeg_alignment_status="aligned", identity_status="consistent",
                                  first_key_segment_sample=first, completion_segment_sample=completion,
                                  completion_kind="space"))
        self.write_fixture()

    def write_fixture(self):
        raw = mne.io.RawArray(self.samples, mne.create_info(["C3", "C4"], 100, "eeg"), verbose=False)
        raw.save(self.root / "typing_raw.fif", overwrite=True, verbose=False)
        with (self.root / "typing_word_timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_real_input_shape_and_shifted_targets(self):
        sequences, summary = load_sequences([self.root], bins=10)
        self.assertEqual(sequences[0].targets, ["prime", "minister"])
        self.assertEqual(sequences[0].features.shape, (2, 40))
        self.assertEqual(summary["examples"], 2)
        example = sequences[0].examples[0]
        self.assertEqual(example["cutoff_segment_sample"], 100)
        self.assertEqual(example["window_start_segment_sample"], 1)
        self.assertLess(example["cutoff_segment_sample"], example["target_first_key_segment_sample"])

    def test_future_eeg_cannot_change_current_input(self):
        before, _ = load_sequences([self.root], bins=10)
        self.samples[:, 101:] += 100
        self.write_fixture()
        after, _ = load_sequences([self.root], bins=10)
        np.testing.assert_array_equal(before[0].features[0], after[0].features[0])
        self.assertFalse(np.array_equal(before[0].features[1], after[0].features[1]))

    def test_target_already_started_is_excluded(self):
        self.rows[1]["first_key_segment_sample"] = 99
        self.write_fixture()
        sequences, summary = load_sequences([self.root], bins=10)
        self.assertEqual(sequences[0].targets, ["minister"])
        self.assertEqual(summary["skipped"]["noncausal_word_order"], 1)

    def test_invalid_alignment_does_not_become_training_data(self):
        self.rows[1]["eeg_alignment_status"] = "unavailable"
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "No causal"):
            load_sequences([self.root])

    def test_duplicate_rows_and_roots_are_rejected(self):
        self.rows.append(self.rows[-1].copy())
        self.write_fixture()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_sequences([self.root])
        with self.assertRaisesRegex(ValueError, "Repeated"):
            load_sequences([self.root, self.root])

    def test_full_cpu_training_and_checkpoint(self):
        dataset = self.root / "export"
        dataset.mkdir()
        rows = []
        for index in range(1, 4):
            filename = f"P{index:03d}_typing_raw.fif"
            raw = mne.io.RawArray(self.samples + index * 1e-7,
                                 mne.create_info(["C3", "C4"], 100, "eeg"), verbose=False)
            raw.save(dataset / filename, overwrite=True, verbose=False)
            rows.extend({**row, "participant_id": f"P{index:03d}", "typing_eeg_file": filename}
                        for row in self.rows)
        with (dataset / "typing_word_timings.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        saved_splits, saved_vocabulary = None, None
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                output = self.root / architecture
                args = build_parser().parse_args(["--data-root", str(dataset), "--output-dir", str(output),
                    "--epochs", "2", "--hidden-dim", "8", "--bins", "4", "--device", "cpu",
                    "--architecture", architecture, "--num-layers", "2"])
                with contextlib.redirect_stdout(io.StringIO()):
                    run(args)
                results = json.loads((output / "metrics.json").read_text())
                self.assertEqual(results["test"]["examples"], 2)
                self.assertEqual(results["test"]["vocabulary_coverage"], 1.0)
                self.assertEqual(results["architecture"], architecture)
                self.assertGreater(results["trainable_parameters"], 0)
                checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["vocabulary"], ["minister", "prime"])
                restored = EEGNextWord(**checkpoint["model_config"]).eval()
                restored.load_state_dict(checkpoint["model_state"])
                example = np.load(output / "training_example.npz", allow_pickle=False)
                self.assertEqual(example["eeg_input"].shape, (1, 2, 16))
                self.assertEqual(example["target_ids"].tolist(), [[1, 0]])
                self.assertEqual(example["target_words"].tolist(), ["prime", "minister"])
                logits = restored(torch.from_numpy(example["eeg_input"]))
                self.assertEqual(tuple(logits.shape), (1, 2, 2))
                self.assertTrue(torch.isfinite(logits).all())
                splits = (output / "splits.json").read_text()
                vocabulary = (output / "vocabulary.json").read_text()
                if saved_splits is not None:
                    self.assertEqual(splits, saved_splits)
                    self.assertEqual(vocabulary, saved_vocabulary)
                saved_splits, saved_vocabulary = splits, vocabulary


class ModelAndSplitTests(unittest.TestCase):
    def sequences(self):
        return [EEGSequence(participant, participant, np.full((2, 4), value, dtype=np.float32),
                            ["prime", "minister"] if participant != "P003" else ["prime", "unseen"],
                            [{"target_word": word} for word in
                             (["prime", "minister"] if participant != "P003" else ["prime", "unseen"])])
                for participant, value in (("P001", 1), ("P002", 2), ("P003", 999))]

    def test_participant_splits_and_train_only_statistics(self):
        splits = split_participants(self.sequences(), 42, ["P002"], ["P003"])
        self.assertEqual([sequence.participant for sequence in splits["train"]], ["P001"])
        vocabulary, counts, mean, scale = fit_training_statistics(splits["train"], 1)
        self.assertNotIn("unseen", vocabulary)
        np.testing.assert_array_equal(mean, np.ones(4))
        with self.assertRaisesRegex(ValueError, "overlap"):
            split_participants(self.sequences(), 42, ["P002"], ["P002"])
        with self.assertRaisesRegex(ValueError, "three participants"):
            split_participants(self.sequences()[:1], 42)

    def test_future_eeg_steps_cannot_change_earlier_predictions(self):
        torch.manual_seed(7)
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = EEGNextWord(4, 3, 8, architecture=architecture, num_layers=2).eval()
                eeg = torch.randn(1, 4, 4)
                before = model(eeg)
                changed = eeg.clone()
                changed[:, 2:] = 1000
                after = model(changed)
                torch.testing.assert_close(before[:, :2], after[:, :2])
                torch.testing.assert_close(before[:, :2], model(eeg[:, :2]))
                self.assertEqual(tuple(before.shape), (1, 4, 3))

    def test_right_padding_does_not_change_valid_predictions(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = EEGNextWord(4, 3, 8, architecture=architecture, num_layers=2).eval()
                eeg = torch.randn(2, 4, 4)
                mask = torch.tensor([[False, False, True, True], [False, False, False, False]])
                batched = model(eeg, padding_mask=mask)
                single = model(eeg[:1, :2])
                torch.testing.assert_close(batched[:1, :2], single)
                torch.testing.assert_close(batched[0, 2:], torch.zeros(2, 3))

    def test_architecture_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "Unknown architecture"):
            EEGNextWord(4, 3, architecture="bidirectional")
        with self.assertRaisesRegex(ValueError, "divisible"):
            EEGNextWord(4, 3, hidden_dim=7, architecture="transformer")
        with self.assertRaisesRegex(ValueError, "positive"):
            EEGNextWord(4, 3, num_layers=0)

    def test_unknown_words_are_misses_not_successful_unknown_class(self):
        sequences = self.sequences()
        vocabulary, counts, mean, scale = fit_training_statistics(sequences[:1], 1)
        dataset = SequenceDataset(sequences[2:], vocabulary, mean, scale)
        batch = collate_sequences([dataset[0]])
        model = EEGNextWord(4, len(vocabulary), 8)
        metrics, predictions = evaluate(model, [batch], torch.device("cpu"), vocabulary, ["prime", "minister"])
        self.assertEqual(metrics["examples"], 2)
        self.assertEqual(metrics["vocabulary_coverage"], 0.5)
        self.assertFalse(predictions[1]["correct"])
        self.assertFalse(predictions[1]["target_in_training_vocabulary"])


if __name__ == "__main__":
    unittest.main()