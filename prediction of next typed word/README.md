# EEG-Only Next Typed Word Prediction

A set of causal baselines: observed EEG windows predict the next typed word.
No previous word IDs, question text, query embeddings, or target text enter the
model. This is recurrent EEG-to-word prediction, not a text-conditioned language
model or unconstrained open-vocabulary generation.

## Real Training Example And Target

The included `example_P012/` contains an actual input from P012, trial 3:

> british prime minister War I

At each step, take the **one second of EEG ending at the current word's following
space**, and predict the next word. Sample indices below are zero-based relative
to that trial's full typing EEG segment, recorded at 512 Hz. Endpoints are inclusive.

| Step | Current Word (Audit Only) | EEG Input Samples | Next-Word Target | Target's First Surviving Key Sample |
| --- | --- | --- | --- | --- |
| 1 | british | 12482 to 12993 | prime | 13596 |
| 2 | prime | 14228 to 14739 | minister | 14927 |
| 3 | minister | 21143 to 21654 | war | 22214 |
| 4 | War | 22926 to 23437 | i | 23733 |

The default GRU carries EEG-derived state forward through this sequence. Thus
the third prediction can depend on EEG features from steps 1, 2, and 3, but not
step 4. No architecture sees strings in the Current Word column.

Each window has 64 channels and 512 samples. It becomes 2,048 EEG features:
64 channels x 16 temporal bins x two statistics (mean amplitude and log variance).
Amplitude is converted to microvolts before feature computation. All mean
features, flattened channel-first/bin-second, precede all log-variance features.
Only samples at or before the cutoff are read; no filtering or resampling is
performed. The GRU operates across word steps, not raw EEG samples.

The complete example input is stored in `example_P012/training_example.npz`:

```python
from pathlib import Path
import numpy as np

example = np.load(
    Path("prediction of next typed word/example_P012/training_example.npz"),
    allow_pickle=False,
)
eeg = example["eeg_input"]
targets = example["target_words"]
print(eeg.shape)       # (1, 4, 2048): batch, word steps, EEG features
print(targets.tolist())  # ['prime', 'minister', 'war', 'i']
```

This inspection example contains unnormalized EEG features. In a training run,
normalization is fitted on training participants only, and saved examples also
contain numeric `target_ids` with shape `[batch, word_steps]`. The target IDs
depend on the training-only vocabulary, so the single-participant inspection
example deliberately does not assign them. Unknown held-out targets use `-1`;
padding uses `-100`.

For a two-word vocabulary `['minister', 'prime']`, the synthetic test example
has input shape `[1, 2, 16]`, targets `['prime', 'minister']`, and target IDs
`[[1, 0]]`. It exercises training, validation, testing, and checkpoint reload.

## Inputs And Installation

Use aligned exports from the updated root `extract_dataset.py`. Each input root
must contain `typing_word_timings.csv` and the linked full typing EEG files.
The older main `dataset/` is not automatically upgraded. Multiple export roots
are accepted, for example separate participant/session exports. Source data is
read-only. Do not duplicate recordings across input roots or participant labels.

```bash
python -m pip install -r "prediction of next typed word/requirements.txt"
```

The system `/usr/bin/python` was used for verification in this workspace; the
workspace virtual environment did not have the EEG dependencies installed.

## Inspect A Dataset

```bash
python "prediction of next typed word/train.py" \
  --data-root /path/to/aligned_export \
  --output-dir "prediction of next typed word/runs/inspection" \
  --inspect-only
```

This writes a coverage summary and one complete query's EEG-feature input and
next-word targets. The verified P012 export yielded **83 next-word examples
across 44 queries**; final words have no next-word target. The included example
was generated from that export; its original absolute EEG paths are provenance,
not portable dataset locations. The saved NPZ can be inspected independently.

## Full Training Command

Training requires **at least three participants** with usable data. The P012-only
example is not enough for an honest participant-held-out experiment.

Run from the repository root, replacing the data path with an aligned export
containing training participants plus P009 and P012:

```bash
python "prediction of next typed word/train.py" \
  --data-root /path/to/aligned_cohort_export \
  --output-dir "prediction of next typed word/runs/next_word_001" \
  --val-participants P009 \
  --test-participants P012 \
  --window-s 1.0 --bins 16 \
  --hidden-dim 128 --batch-size 16 \
  --epochs 50 --patience 8 \
  --learning-rate 0.001 --seed 42 --device auto
```

All other participants are training participants. Alternatively omit both
held-out participant arguments for a seeded approximately 60/20/20 split, with
at least one participant in each set. `--data-root` can take multiple roots.
Use a new or empty output directory outside the input roots. `--device cpu`
forces CPU execution; `auto` selects CUDA when available.

Every variant uses the same EEG feature projection and word-classification
objective. Query trials are independent; recurrent state and positional indices
reset between them. Query sequences may be shuffled; their internal word order
is preserved. Training uses next-word cross-entropy, AdamW, gradient clipping,
and validation-loss early stopping. No teacher-forced text tokens enter any model.

## Model Variants

Select a model with `--architecture`:

| Architecture | EEG History | Causality Mechanism |
| --- | --- | --- |
| `mlp` | Current window only | Each word step is processed independently |
| `gru` (default) | Current and all earlier steps | Unidirectional gated recurrent state |
| `lstm` | Current and all earlier steps | Unidirectional hidden and cell state |
| `tcn` | Current and a finite number of earlier steps | Dilated convolutions padded on the left only |
| `transformer` | Current and all earlier steps | Explicit upper-triangular attention mask plus padding mask |

The MLP is an ablation: it tests whether sequence history helps compared with
just the latest EEG window. The TCN convolves across word-step feature vectors,
not individual raw EEG samples. It never uses centered convolutions or
normalization across time. Transformer inputs have sinusoidal positional
encodings, and future keys are blocked at every attention layer. Right-padding
is masked explicitly; unknown words are not mistaken for padded positions.

`--num-layers` controls GRU/LSTM layers, TCN blocks, or Transformer layers and
defaults to 1. It is ignored by the MLP. `--kernel-size` controls the TCN kernel
(default 3); its receptive field is `1 + (kernel_size - 1) * (2**num_layers - 1)`
word steps. `--attention-heads` controls Transformer attention heads (default 4);
the hidden dimension must be divisible by this value.

For example, add these options to the training command above:

```bash
--architecture transformer --num-layers 2 --attention-heads 4
```

To run all five on the same data, held-out participants, and seed:

```bash
for architecture in mlp gru lstm tcn transformer; do
  python "prediction of next typed word/train.py" \
    --data-root /path/to/aligned_cohort_export \
    --output-dir "prediction of next typed word/runs/${architecture}_seed42" \
    --val-participants P009 --test-participants P012 \
    --architecture "$architecture" --num-layers 2 \
    --hidden-dim 128 --epochs 50 --patience 8 --seed 42
done
```

All runs use the same vocabulary, feature normalization, and evaluation rules.
Their parameter counts differ and are recorded in configuration and metrics;
equal hidden dimensions do not imply equal model capacity. Select architecture
and hyperparameters using validation results, not repeated test-set comparisons.
Use multiple seeds before drawing conclusions about which model performs best.

Outputs include:

- `best_model.pt`: best-validation model, architecture parameters, training
  normalization, vocabulary, and configuration; load with `weights_only=True`.
  Reconstruct with `EEGNextWord(**checkpoint["model_config"])`, then load
  `checkpoint["model_state"]`. This preserves architecture-specific settings.
- `metrics.json`: final train/validation/test results from the best checkpoint.
- `history.json`: epoch-level training and validation losses/metrics.
- `splits.json`, `vocabulary.json`, `config.json`, `data_summary.json`: provenance,
  participant assignment, preprocessing settings, and excluded-pair counts.
- `training_example.npz` and `training_example.json`: one training sequence's
  normalized EEG input, labels, and source/cutoff audit records.
- `*_predictions.jsonl`: per-example predictions, top-five probabilities, true
  targets, vocabulary coverage flags, and EEG-window provenance.

## Validity And Interpretation

Only `csv_validated`, EEG-`aligned` rows with consistent or manifest-confirmed
participant identity are eligible. Adjacent word indices must be consecutive.
EEG channels must match by name and order. Missing or unreliable boundaries,
short windows, and nonfinite EEG are excluded. `completion` uses actual following
spaces only; `--boundary last_key` instead cuts at the latest surviving character.
The target's first surviving key must fall strictly after the cutoff. Routine-end
proxies and missing full-segment mappings are not silently substituted.

Vocabulary and feature normalization use training data only. Validation selects
the checkpoint; test data does not. Unknown targets are excluded from supervised
loss but count as errors in all-target accuracy. Reports include vocabulary
coverage, known-target loss/accuracy, top-1/top-5 all-target accuracy, and a
training-frequency baseline. Top-five accuracy is trivial with five or fewer
classes, so always inspect vocabulary size and the baseline.

This is an exploratory baseline, not evidence that EEG decodes intended words.
Word boundaries are offline annotations derived from typing, not an EEG-only
boundary detector. Labels/timestamps describe the final surviving text; deleted
keystrokes and editing can make inferred intention or first-ever word entry
ambiguous. EEG includes motor, ocular, and muscle signals. Any preprocessing
already applied to source EEG must be checked for future-sample leakage.
Participant-held-out evaluation does not guarantee unseen questions or queries;
repeated prompts and duplicate recordings require a separate leakage audit.
The model cannot predict words absent from its training vocabulary, or generate
multiple future words without additional observed EEG steps.

## Tests

```bash
python -m unittest discover -s "prediction of next typed word" \
  -p test_next_word.py -v
```

Tests include future-sample invariance, causal/prefix-consistent predictions and
padding invariance for all five architectures, train-only vocabulary and
normalization, participant-disjoint splits, and unknown-word scoring. Every
variant completes a two-epoch CPU training test and architecture-aware checkpoint
reload, with identical splits and vocabularies. Synthetic test accuracy is not a
scientific result. No full-cohort model has been trained yet.