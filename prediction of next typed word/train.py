"""Train a causal EEG-only next-typed-word baseline on aligned typing exports."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from data import EEGSequence, load_sequences
from model import ARCHITECTURES, EEGNextWord


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def split_participants(sequences: list[EEGSequence], seed: int,
                       validation: list[str] | None = None, test: list[str] | None = None) -> dict:
    participants = sorted({sequence.participant for sequence in sequences})
    if len(participants) < 3:
        raise ValueError("Training requires at least three participants for disjoint train/validation/test sets; use --inspect-only for a single-participant export")
    if (validation is None) != (test is None):
        raise ValueError("Specify both --val-participants and --test-participants, or neither")
    if validation is None:
        shuffled = participants.copy()
        random.Random(seed).shuffle(shuffled)
        held_out = max(1, len(participants) // 5)
        test, validation = shuffled[:held_out], shuffled[held_out:2 * held_out]
    if set(validation) & set(test):
        raise ValueError("Validation and test participants overlap")
    if (set(validation) | set(test)) - set(participants):
        raise ValueError("Requested held-out participant has no usable EEG examples")
    training = sorted(set(participants) - set(validation) - set(test))
    if not training or not validation or not test:
        raise ValueError("All three participant splits must be nonempty")
    ids = {"train": training, "validation": sorted(validation), "test": sorted(test)}
    return {name: [sequence for sequence in sequences if sequence.participant in selected]
            for name, selected in ids.items()}


def fit_training_statistics(sequences: list[EEGSequence], min_word_count: int):
    counts = Counter(word for sequence in sequences for word in sequence.targets)
    vocabulary = sorted(word for word, count in counts.items() if count >= min_word_count)
    if len(vocabulary) < 2:
        raise ValueError("Training needs at least two target word classes; lower --min-word-count or supply more data")
    features = np.concatenate([sequence.features for sequence in sequences]).astype(np.float64)
    mean = features.mean(axis=0).astype(np.float32)
    scale = np.maximum(features.std(axis=0), 1e-6).astype(np.float32)
    return vocabulary, counts, mean, scale


class SequenceDataset(Dataset):
    def __init__(self, sequences, vocabulary, mean, scale):
        self.sequences = sequences
        self.word_ids = {word: index for index, word in enumerate(vocabulary)}
        self.mean, self.scale = mean, scale

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence = self.sequences[index]
        eeg = np.clip((sequence.features - self.mean) / self.scale, -20, 20)
        targets = [self.word_ids.get(word, -1) for word in sequence.targets]
        return torch.from_numpy(eeg), torch.tensor(targets, dtype=torch.long), sequence.examples


def collate_sequences(batch):
    eeg, targets, examples = zip(*batch)
    return pad_sequence(eeg, batch_first=True), pad_sequence(targets, batch_first=True, padding_value=-100), examples


def evaluate(model, loader, device, vocabulary, frequency_words):
    model.eval()
    totals = Counter()
    predictions = []
    with torch.no_grad():
        for eeg, targets, examples in loader:
            targets = targets.to(device)
            logits = model(eeg.to(device), padding_mask=targets.eq(-100))
            known = targets >= 0
            if known.any():
                totals["nll_sum"] += functional.cross_entropy(logits[known], targets[known], reduction="sum").item()
            probabilities = logits.softmax(dim=-1)
            values, indices = probabilities.topk(min(5, len(vocabulary)), dim=-1)
            values, indices = values.cpu(), indices.cpu()
            for batch_index, records in enumerate(examples):
                for step, record in enumerate(records):
                    target_id = int(targets[batch_index, step].item())
                    top_ids = indices[batch_index, step].tolist()
                    top_words = [vocabulary[index] for index in top_ids]
                    is_known = target_id >= 0
                    correct = is_known and top_ids[0] == target_id
                    totals["examples"] += 1
                    totals["known"] += is_known
                    totals["correct"] += correct
                    totals["top5"] += is_known and target_id in top_ids
                    totals["baseline_correct"] += record["target_word"] == frequency_words[0]
                    totals["baseline_top5"] += record["target_word"] in frequency_words[:5]
                    predictions.append({**record, "target_in_training_vocabulary": is_known,
                                        "predicted_word": top_words[0], "correct": bool(correct),
                                        "top_words": top_words, "top_probabilities": values[batch_index, step].tolist()})
    count, known = totals["examples"], totals["known"]
    metrics = {"examples": count, "known_target_examples": known,
               "vocabulary_coverage": known / count if count else 0.0,
               "top1_accuracy_all": totals["correct"] / count if count else 0.0,
               "top5_accuracy_all": totals["top5"] / count if count else 0.0,
               "top1_accuracy_known": totals["correct"] / known if known else None,
               "nll_known": totals["nll_sum"] / known if known else None,
               "frequency_baseline_top1": totals["baseline_correct"] / count if count else 0.0,
               "frequency_baseline_top5": totals["baseline_top5"] / count if count else 0.0}
    return metrics, predictions


def save_example(output, sequence, mean=None, scale=None, vocabulary=None):
    features = sequence.features.copy()
    if mean is not None:
        features = np.clip((features - mean) / scale, -20, 20)
    arrays = {"eeg_input": features[None], "target_words": np.array(sequence.targets)}
    description = {"input_shape": list(arrays["eeg_input"].shape),
                   "axes": ["batch", "causal_word_steps", "EEG_features"],
                   "normalized_with_training_statistics": mean is not None,
                   "targets_are_model_inputs": False,
                   "steps": sequence.examples}
    if vocabulary is not None:
        word_ids = {word: index for index, word in enumerate(vocabulary)}
        arrays["target_ids"] = np.array([[word_ids.get(word, -1) for word in sequence.targets]])
        description["target_ids"] = arrays["target_ids"].tolist()
    np.savez_compressed(output / "training_example.npz", **arrays)
    write_json(output / "training_example.json", description)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, nargs="+", required=True,
                        help="One or more extract_dataset.py export roots containing typing_word_timings.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--boundary", choices=["completion", "last_key"], default="completion")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="gru")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Recurrent, temporal CNN, or Transformer depth; ignored by mlp")
    parser.add_argument("--attention-heads", type=int, default=4,
                        help="Transformer heads; hidden-dim must be divisible by this")
    parser.add_argument("--kernel-size", type=int, default=3, help="Causal temporal CNN kernel size")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--min-word-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--val-participants", nargs="+")
    parser.add_argument("--test-participants", nargs="+")
    return parser


def run(args):
    for name in ("epochs", "patience", "batch_size", "hidden_dim", "min_word_count",
                 "num_layers", "attention_heads", "kernel_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be nonnegative and finite")
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new or empty output directory")
    if any(output == root.expanduser().resolve() or root.expanduser().resolve() in output.parents
           or output in root.expanduser().resolve().parents for root in args.data_root):
        raise ValueError("Outputs must be separate from input datasets")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sequences, summary = load_sequences(args.data_root, args.window_s, args.bins, args.boundary)
    if args.inspect_only:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "data_summary.json", summary)
        save_example(output, sequences[0])
        print(json.dumps(summary, indent=2))
        print(f"EEG input and word targets saved in {output}")
        return
    splits = split_participants(sequences, args.seed, args.val_participants, args.test_participants)
    vocabulary, counts, mean, scale = fit_training_statistics(splits["train"], args.min_word_count)
    frequency_words = sorted(vocabulary, key=lambda word: (-counts[word], word))
    loaders = {name: DataLoader(SequenceDataset(selected, vocabulary, mean, scale),
                               batch_size=args.batch_size, shuffle=name == "train", collate_fn=collate_sequences)
               for name, selected in splits.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model_config = {"feature_dim": summary["feature_dim"], "vocabulary_size": len(vocabulary),
                    "hidden_dim": args.hidden_dim, "architecture": args.architecture,
                    "num_layers": args.num_layers, "attention_heads": args.attention_heads,
                    "kernel_size": args.kernel_size}
    model = EEGNextWord(**model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    validation_metrics, _ = evaluate(model, loaders["validation"], device, vocabulary, frequency_words)
    if validation_metrics["known_target_examples"] == 0:
        raise ValueError("Validation has no targets in the training vocabulary; supply more data or change the participant split")
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "data_root": [str(path.expanduser().resolve()) for path in args.data_root],
              "output_dir": str(output), "device": str(device),
              "trainable_parameters": parameter_count,
              "torch_version": str(torch.__version__), "numpy_version": str(np.__version__)}
    write_json(output / "config.json", config)
    write_json(output / "data_summary.json", summary)
    write_json(output / "splits.json", {name: {"participants": sorted({sequence.participant for sequence in selected}),
                                              "sequences": len(selected), "examples": sum(len(sequence.targets) for sequence in selected)}
                                       for name, selected in splits.items()})
    write_json(output / "vocabulary.json", {"words": vocabulary, "training_counts": dict(counts),
                                           "oov_target_id": -1, "padding_target_id": -100})
    save_example(output, splits["train"][0], mean, scale, vocabulary)
    history, best_loss, stale = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, target_count = 0.0, 0
        for eeg, targets, examples in loaders["train"]:
            eeg, targets = eeg.to(device), targets.to(device)
            known = targets >= 0
            if not known.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg, padding_mask=targets.eq(-100))
            loss = functional.cross_entropy(logits[known], targets[known])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_count = int(known.sum().item())
            loss_sum += loss.item() * batch_count
            target_count += batch_count
        validation_metrics, _ = evaluate(model, loaders["validation"], device, vocabulary, frequency_words)
        item = {"epoch": epoch, "train_nll_known": loss_sum / target_count, "validation": validation_metrics}
        history.append(item)
        write_json(output / "history.json", history)
        print(json.dumps(item), flush=True)
        if validation_metrics["nll_known"] < best_loss:
            best_loss, stale = validation_metrics["nll_known"], 0
            torch.save({"model_state": model.state_dict(), "feature_dim": summary["feature_dim"],
                        "hidden_dim": args.hidden_dim, "vocabulary": vocabulary,
                        "model_config": model_config,
                        "normalization_mean": torch.from_numpy(mean), "normalization_scale": torch.from_numpy(scale),
                        "data_summary": summary, "config": config, "best_epoch": epoch}, output / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    results = {"best_epoch": checkpoint["best_epoch"], "architecture": args.architecture,
               "trainable_parameters": parameter_count}
    for name, loader in loaders.items():
        metrics, predictions = evaluate(model, loader, device, vocabulary, frequency_words)
        results[name] = metrics
        with (output / f"{name}_predictions.jsonl").open("w", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, allow_nan=False) + "\n")
    write_json(output / "metrics.json", results)
    print(json.dumps(results, indent=2))


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Training setup failed: {error}\n")


if __name__ == "__main__":
    main()