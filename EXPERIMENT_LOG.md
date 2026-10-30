# Experiment Log

## 2026-10-20 13:00 BST (+0100): Task EEG Query-to-Response BPR Ablations

### Correction to the previous experiment description

The earlier Task BPR runs did **not** test the primary research question: whether EEG recorded while a participant reads a query can retrieve the response shown for that query. They instead used two separate pair types:

- Question-reading EEG from a sampled question span -> the full question text.
- Response-reading EEG from a sampled displayed-response span -> that displayed response.

I incorrectly described those runs as answering query-to-response retrieval. They did not. The required primary pair is question-reading EEG -> the exact `response_seen` text for the same trial, used as the positive document. It reuses the question-reading EEG query representation; response-reading EEG is a separate auxiliary source. `ground_truth_response` is not substituted for the displayed response.

### Data prepared

The Task builder now writes three pair sources and a union:

- `authentic_query_response`: sampled question-span EEG -> same-trial `response_seen`.
- `authentic_question_ict`: sampled question-span EEG -> full question.
- `authentic_response_ict`: sampled displayed-response-span EEG -> full `response_seen`.

Full files retain the explicit unverified-offset P001/P002 samples for sensitivity analysis. Separate `_aligned_pairs.npy` files filter to `alignment_status=aligned_word_index` and are the only Task files used by the primary jobs.

| Pair source | Full pairs | Aligned primary pairs |
|---|---:|---:|
| Query EEG -> displayed response | 1,410 | 1,328 |
| Question ICT | 1,410 | 1,328 |
| Response ICT | 2,028 | 1,860 |

The aligned-only counts reflect exclusion of unverified offset EEG. Source manifests retain alignment and generation metadata.

### Experimental design

All seven arms train on the authentic query-to-response source. The auxiliary Task sources and Alice/Nieuwland are added one at a time or together:

| Condition | Training sources | Slurm job |
|---|---|---:|
| Query-response baseline | Query-response | 8208840 |
| Add question ICT | Query-response + question ICT | 8208841 |
| Add response ICT | Query-response + response ICT | 8208842 |
| Full Task | Query-response + both ICT sources | 8208836 |
| Full Task + Audio | Full Task + Alice ICT | 8208837 |
| Full Task + Visual | Full Task + Nieuwland ICT | 8208839 |
| Full Task + Audio + Visual | Full Task + Alice + Nieuwland ICT | 8208838 |

Normalized question text determines the deterministic 80/10/10 split globally across participants and Task pair types. The same seed (42) and source-independent hash assignment make the primary Task train/validation/test query set identical across ablations. Checkpoint selection and early stopping use only held-out `authentic_query_response` validation. Final test evaluation uses its same-source held-out response candidate pool, with MRR/Recall@k and Gaussian-noise baseline; source-specific auxiliary metrics are also logged.

Question ICT and query-response pairs for the same question EEG span share a query-group ID. If both appear in an in-batch loss, both their full-question and displayed-response documents count as positives, not negatives. Task types and external sources are balanced in mixed training.

### Model and training settings

- Dual encoder; simple EEG encoder, hidden dimension 768; mean pooling.
- Frozen pretrained ColBERT-v2 text encoder; LoRA disabled.
- Batch size 32; AdamW learning rate `1e-5`; max 200 epochs; early-stopping patience 5.
- Seed 42; training masking 90%; validation masking at 0/25/50/75/90/100% for ICT. The primary query-to-response target remains unmasked.
- One GPU, 12 CPUs, 128 GB RAM, 24-hour time limit; `gpu04` excluded.
- W&B project: `Authentic Queries BPR`; run names use `aqbpr_<condition>_mean_seed42_dynamic_mask` and log condition, source set, input files, split, model, and primary evaluation source.

### Submission and correction history

Two earlier pending batches were cancelled before any job started:

- `8208817`-`8208823`: cancelled after finding that shuffling only the groups present in each ablation could change the primary split when an auxiliary source was added.
- `8208826`-`8208832`: cancelled after noticing that the primary analysis was supposed to exclude P001/P002's unverified EEG alignment.

The final aligned-only batch above was submitted on 2026-10-20. At log time, all seven jobs were pending for GPU resources; no performance results are available yet. The corrected matrix and job metadata are in `Audio_BPR/prepared_experiments_20261020_query_response_ablations/experiment_matrix.json`.

### Verification

All 11 Audio_BPR smoke tests passed after the pair builder, multi-positive objective, source-independent split, primary validation selection, W&B naming, and aligned-only file selection changes.

## 2026-10-26: Audio BPR Protocol Runs (Original Random Split)

### Why the setup changed

The question-held-out, target-grouped, trial-grouped and per-source pair splits used between 2026-10-20 and 2026-10-22 were not the Audio BPR protocol. Their results (noise-level query→response MRR) are superseded. All runs below reuse the Audio BPR non-holdout setup exactly, with our Task data added as extra sources.

### Data (v4, aligned participants only; P001/P002 excluded)

| Source | Query EEG | Positive document | Aligned pairs | Masked at test? |
|---|---|---|---:|---|
| `authentic_query_response` | All question fixations for the trial (full question read) | Same-trial `response_seen` | 691 | No |
| `authentic_question_ict` | Question span | Full question (span removed when masked) | 1,328 | Yes |
| `authentic_response_ict` | Displayed-response span | Full response | 1,860 | No |
| Alice (audio) | ICT span | Masked sentence context | 8,134 | Yes |
| Nieuwland (visual) | ICT span | Masked sentence context | 8,132 | Yes |

25 aligned participants, 48 question IDs and 48 responses.

### Protocol (same as Audio BPR when `--holdout_subjects` is off)

- Split (`--split-strategy original`): all loaded pairs are pooled and shuffled with `random.seed(42)`. Train is the first 80%, validation is the remaining 20%, and test is the second half of validation (validation and test overlap, as in the original). The split is pair-level, so participants and trials appear in more than one split.
- Index: test documents are de-duplicated by masked text, with one correct document per query. The index grows with masking.
- Testing: one `test_model` pass over the full test set at masking 0/25/50/75/90/100, with per-dataset ranking (100-document subsets) and a Gaussian-noise baseline at 90% masking.
- Early stopping: on the full validation loss.
- Model and training: dual encoder; simple EEG encoder (hidden 768); mean pooling; frozen pretrained ColBERT-v2 (no LoRA); batch size 32; AdamW lr 1e-5; up to 200 epochs; patience 5; seed 42; 90% training masking.
- Deviations from the original:
  - Multi-positive in-batch loss: duplicate responses are not treated as negatives.
  - Validation ranking for mixed sources is done per source only (this fixed an out-of-memory error).
  - Extra metrics are saved per source and against a 95-document question+response index. These do not affect the main results.
- Compute: one GPU, 12 CPUs, 24 h, `gpu04` excluded. Memory was 128 GB, except arm D, which used 256 GB.
- W&B: project `Authentic Queries BPR`, run names `aqbpr_<arm>_mean_seed42_originalsplit`.

Cancelled before starting: `8219237`-`8219243` (trial split) and `8219341`-`8219347` (subject holdout; we decided not to use holdout yet).

### Batch 1: Task ablations (`Audio_BPR/prepared_experiments_20261022_audio_bpr_originalsplit`, jobs `8219348`-`8219354`)

Results on the full pooled test index:

| Arm | Test queries | Docs (90%) | MRR 0% | MRR 90% | Noise MRR | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| query_response_only | 70 | 35 | 0.134 | 0.134 | 0.132 | 3 |
| + question_ict | 202 | 143 | 0.233 | 0.166 | 0.047 | 71 |
| + response_ict | 256 | 187 | 0.142 | 0.042 | 0.031 | 7 |
| all_task_sources | 388 | 292 | 0.113 | 0.046 | 0.023 | 23 |
| all_task + audio | 1202 | 903 | 0.352 | 0.148 | 0.008 | 50 |
| all_task + visual | 1202 | 906 | 0.286 | 0.120 | 0.009 | 8 |
| all_task + audio + visual | — | — | — | — | — | out of memory (128 GB) |

Findings:

- Query→response alone does not learn: there are only about 570 training pairs.
- Adding question ICT gives the best Task result.
- Response ICT pulls query→response down to noise level when the index includes masked ICT documents. Its query→response MRR against the full index at 90% masking is 0.024 (noise 0.026). Likely causes:
  - Its documents are unmasked full responses, so it is not true ICT.
  - It contributes 1,860 pairs that all point at only 48 texts.

### Batch 2: query→response + question ICT with external data, no response ICT (jobs `8225508`-`8225510`)

| Arm | Sources | Job | Output |
|---|---|---|---|
| A | query→response + question ICT | `8219353` (from batch 1) | `prepared_experiments_20261022_audio_bpr_originalsplit` |
| B | A + Alice | `8225508` | `prepared_experiments_20261023_qr_qict_modalities` |
| C | A + Nieuwland | `8225509` | `prepared_experiments_20261023_qr_qict_modalities` |
| D | A + Alice + Nieuwland | `8225510` (256 GB) | `prepared_experiments_20261023_qr_qict_modalities_256G` |

All three batch-2 jobs completed. Best epochs: A 71, B 54, C 7, D 29.

Full pooled test index (all test queries against all indexed test documents). MRR:

| Arm | 0% | 25% | 50% | 75% | 90% | 100% | Noise (90%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: query→response + question ICT | 0.233 | 0.212 | 0.172 | 0.165 | 0.166 | 0.167 | 0.047 |
| B: A + audio | 0.381 | 0.266 | 0.204 | 0.169 | 0.166 | 0.173 | 0.010 |
| C: A + visual | 0.295 | 0.201 | 0.159 | 0.136 | 0.123 | 0.127 | 0.008 |
| D: A + audio + visual | 0.350 | 0.237 | 0.181 | 0.152 | 0.152 | 0.158 | 0.005 |

Hit@10:

| Arm | 0% | 25% | 50% | 75% | 90% | 100% | Noise (90%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A: query→response + question ICT | 0.416 | 0.376 | 0.347 | 0.342 | 0.347 | 0.337 | 0.074 |
| B: A + audio | 0.601 | 0.446 | 0.383 | 0.350 | 0.343 | 0.345 | 0.011 |
| C: A + visual | 0.395 | 0.310 | 0.278 | 0.259 | 0.251 | 0.252 | 0.011 |
| D: A + audio + visual | 0.515 | 0.391 | 0.342 | 0.317 | 0.301 | 0.303 | 0.007 |

Index size by arm:

| Arm | Test queries | Indexed docs, 0% → 90% |
|---|---:|---|
| A | 202 | 84 → 143 |
| B | 1016 | 161 → 761 |
| C | 1016 | 158 → 757 |
| D | 1829 | 240 → 1368 |

Index sizes differ, so compare each row with its own noise baseline.

Per source, with each source's queries ranked against its own documents (MRR at 90% masking, then noise):

| Source | A | B | C | D |
|---|---|---|---|---|
| Query→response | 0.293 / 0.129 | 0.188 / 0.154 | 0.110 / 0.105 | 0.134 / 0.127 |
| Question ICT | 0.208 / 0.064 | 0.099 / 0.048 | 0.048 / 0.043 | 0.082 / 0.047 |
| Alice | — | 0.188 / 0.013 | — | 0.152 / 0.012 |
| Nieuwland | — | — | 0.145 / 0.010 | 0.173 / 0.009 |

Findings:

- Every arm is well above noise on the full pooled index.
- Arm A remains the best for the Task sources. Adding Alice or Nieuwland lowers the per-source query→response and question ICT results, and with visual both fall to noise level. The larger external sources (about 6,500 training pairs each, against roughly 1,600 Task pairs) likely swamp the Task signal, since sources are not balanced.

Caveats:

- Single seed.
- Only 51–76 query→response test queries per arm.
- Test sets differ between arms because the original split reshuffles the combined pool.
- The pair-level split can put question-ICT spans from a trial in training while that trial's query→response pair is in test.

## 2026-10-27: Typing EEG Runs, Training Dynamics and Planned Changes

### Typing data and runs

- Pairs are built by `Audio_BPR/Dataset/authentic_typing_reader.py` into `dataset/authentic_typing_pairs/`. The data covers 882 validated typing trials from 23 participants. P003 and P007 are excluded because they have no keystroke log.
- Two EEG formats are built for the same trials and spans:
  - Continuous: 1 s windows, from 200 ms before the first logged key to 800 ms after the last.
  - Keypress: one −200 to 800 ms epoch per keypress that survives into the submitted query.
- Pair types:
  - Typing → response: the full typing EEG, with the same trial's `response_seen` as the positive.
  - Typing ICT: the EEG for a typed span, with the submitted typed query as the positive (span masked at runtime).
- Protocol: the same as the reading runs (original Audio BPR split, seed 42, 90% training masking, early stopping on validation loss over all sources).
- Jobs: continuous A–D `8236686`, `8236687`, `8236688`, `8236692`; keypress A–D `8236689`, `8236690`, `8236691`, `8236693`. Arms B and C add Alice and Nieuwland respectively; arm D adds both and uses 256 GB.
- Loss plots: `findings/training_validation_losses_typing_{continuous,keypress}.png`.

### Signal above noise

- Continuous typing alone is above noise at every masking level. At 90% masking on the full index, MRR is 0.060 (noise 0.025) and Hit@10 is 0.105 (noise 0.043).
- Keypress typing → response gives the best single typing result: 0.145 vs 0.088 noise. With audio + visual added it is 0.142 vs 0.102.
- Continuous typing ICT alone: 0.056 vs 0.035 noise.

### Training dynamics (loss curves and W&B `train/grad_norm`)

- Every step of every run is clipped. The clip threshold is 1.0 and pre-clip gradient norms are about 3–28. The reading runs show the same.
- Continuous typing alone is still learning when it stops: gradient norms rise from 4 to 13, as they did in reading arm A (5 to 17). Train and validation losses stay close (gap 0.36), so it is underfitting.
- Keypress typing alone ran only 384 steps (6 epochs), with flat gradient norms around 3. Patience 5 stopped it before learning took off.
- Adding visual makes the model memorise. Gradient norms spike mid-run (about 25) and then collapse (about 6), training loss drops below 0.5, and validation loss rises after epoch 7–9.
- Adding audio gives the most stable training. Gradient norms peak and then settle (about 28 to 20), and validation loss improves until about epoch 50.
- In the mixed arms, early stopping and checkpoint choice follow Alice and Nieuwland, because they dominate the validation loss.

### Planned changes for the next typing round

1. Raise the keypress sequence limit from 50 to about 100, so the 18% of keypress → response pairs currently cut at 50 keys keep the full query.
2. Add a word-level continuous format: one window per typed word, from its first to its last key.
3. Use the displayed question as the typing ICT target, so targets are shared across participants. Each typed query is phrased differently, so the current targets are almost all unique.
4. Pretrain on Alice + Nieuwland, then fine-tune on typing, or use source balancing, so the larger external sets don't dominate.
5. Run 3 seeds of the typing-only arms to confirm the signal above noise.
6. For typing-only runs, use patience 15–20 (or a minimum number of epochs), clip threshold 5, and try learning rate 3e-5.
7. For mixed arms, use the typing validation loss for early stopping.

## 2026-10-28: Typing v2 Runs

### What changed from typing v1

The changes below apply only to the `typing_v2_*` arms; reading arms keep the original Audio BPR flags.

- New word-level EEG format: one −200 to 800 ms epoch per typed word, starting at the word's first surviving key. It sits alongside the continuous and keypress formats.
- Typing ICT now uses the displayed question as its target, with the typed span masked. Only spans typed verbatim in the question are used (789 of 1,676); the rest were reworded or misspelled.
- Typing-only training settings:
  - Maximum sequence length 100 (was 50). The 161 keypress pairs cut at 50 keys are now complete.
  - Learning rate 3e-5.
  - Patience 20, with no early stopping before epoch 10.
  - Gradient clip 5.0.
  - Early stopping on the typing validation loss only.
- Arms with Alice or Nieuwland use source balancing.
- The typing-only arms were run with 3 seeds (42, 43, 44).
- Code: `Audio_BPR` commit `4dd47ac`. Jobs `8237640`–`8237657`, with results in `Audio_BPR/prepared_experiments_20261027_typing_v2{,_256G}/`. The before/after figure is `findings/typing_formats_before_after.png`.

### Full pooled test index, MRR

Typing-only rows are the mean ± SD over 3 seeds; the external-data rows are seed 42. Typing-only arms have 168 test queries, with 76 indexed documents at 0% masking rising to 107 at 90%.

| Arm | 0% | 25% | 50% | 75% | 90% | 100% | Noise |
|---|---|---|---|---|---|---|---|
| Continuous | 0.103±.006 | 0.093±.004 | 0.087±.004 | 0.084±.004 | 0.085±.006 | 0.086±.007 | 0.040 |
| Keypress | 0.087±.002 | 0.078±.002 | 0.081±.002 | 0.078±.004 | 0.078±.003 | 0.078±.002 | 0.043 |
| Word | 0.074±.010 | 0.064±.010 | 0.055±.010 | 0.053±.008 | 0.052±.008 | 0.052±.008 | 0.049 |
| Continuous + audio | 0.288 | 0.178 | 0.143 | 0.119 | 0.113 | 0.119 | 0.009 |
| Continuous + visual | 0.308 | 0.204 | 0.160 | 0.148 | 0.135 | 0.143 | 0.009 |
| Continuous + audio + visual | 0.329 | 0.210 | 0.168 | 0.149 | 0.143 | 0.150 | 0.006 |
| Keypress + audio | 0.269 | 0.166 | 0.128 | 0.117 | 0.102 | 0.110 | 0.009 |
| Keypress + visual | 0.300 | 0.204 | 0.155 | 0.145 | 0.132 | 0.139 | 0.010 |
| Keypress + audio + visual | 0.311 | 0.193 | 0.158 | 0.140 | 0.138 | 0.143 | 0.005 |
| Word + audio | 0.277 | 0.167 | 0.142 | 0.115 | 0.109 | 0.115 | 0.009 |
| Word + visual | 0.305 | 0.201 | 0.156 | 0.142 | 0.127 | 0.139 | 0.012 |
| Word + audio + visual | 0.319 | 0.208 | 0.163 | 0.146 | 0.139 | 0.142 | 0.005 |

### Full pooled test index, Hit@10

| Arm | 0% | 25% | 50% | 75% | 90% | 100% | Noise |
|---|---|---|---|---|---|---|---|
| Continuous | 0.242 | 0.220 | 0.193 | 0.191 | 0.202 | 0.201 | 0.069 |
| Keypress | 0.190 | 0.181 | 0.157 | 0.157 | 0.165 | 0.163 | 0.070 |
| Word | 0.185 | 0.165 | 0.129 | 0.125 | 0.129 | 0.127 | 0.107 |
| Continuous + audio | 0.557 | 0.384 | 0.302 | 0.263 | 0.253 | 0.260 | 0.011 |
| Continuous + visual | 0.401 | 0.325 | 0.294 | 0.273 | 0.271 | 0.275 | 0.013 |
| Continuous + audio + visual | 0.516 | 0.367 | 0.321 | 0.293 | 0.283 | 0.285 | 0.006 |
| Keypress + audio | 0.521 | 0.338 | 0.278 | 0.245 | 0.231 | 0.238 | 0.013 |
| Keypress + visual | 0.405 | 0.321 | 0.286 | 0.283 | 0.271 | 0.271 | 0.013 |
| Keypress + audio + visual | 0.479 | 0.351 | 0.305 | 0.285 | 0.265 | 0.273 | 0.007 |
| Word + audio | 0.546 | 0.353 | 0.291 | 0.254 | 0.235 | 0.248 | 0.012 |
| Word + visual | 0.406 | 0.317 | 0.290 | 0.280 | 0.269 | 0.270 | 0.016 |
| Word + audio + visual | 0.501 | 0.354 | 0.308 | 0.288 | 0.273 | 0.277 | 0.006 |

### Per source, MRR at 90% masking / noise

The typing → response pool has about 40 responses; the question ICT pool has 67–75 questions. Typing-only values are 3-seed means.

| Arm | Typing → response | Typing question ICT | Alice | Nieuwland |
|---|---|---|---|---|
| Continuous | 0.122 / 0.097 | 0.074 / 0.068 | — | — |
| Keypress | 0.137 / 0.107 | 0.109 / 0.067 | — | — |
| Word | 0.103 / 0.093 | 0.074 / 0.064 | — | — |
| Continuous + audio | 0.107 / 0.096 | 0.075 / 0.070 | 0.122 / 0.010 | — |
| Continuous + visual | 0.131 / 0.091 | 0.091 / 0.082 | — | 0.146 / 0.010 |
| Continuous + audio + visual | 0.140 / 0.092 | 0.092 / 0.058 | 0.123 / 0.009 | 0.173 / 0.010 |
| Keypress + audio | 0.120 / 0.110 | 0.068 / 0.061 | 0.112 / 0.011 | — |
| Keypress + visual | 0.108 / 0.115 | 0.111 / 0.062 | — | 0.145 / 0.011 |
| Keypress + audio + visual | 0.101 / 0.086 | 0.146 / 0.062 | 0.126 / 0.012 | 0.162 / 0.011 |
| Word + audio | 0.097 / 0.111 | 0.098 / 0.067 | 0.118 / 0.010 | — |
| Word + visual | 0.110 / 0.120 | 0.094 / 0.056 | — | 0.139 / 0.013 |
| Word + audio + visual | 0.113 / 0.089 | 0.079 / 0.062 | 0.115 / 0.015 | 0.180 / 0.013 |

### Findings

- The three seeds agree closely. Continuous typing alone scores 0.085 ± 0.006 against 0.040 noise at 90% masking (Hit@10 0.20 vs 0.07).
- Keypress typing ICT against the displayed question is the most consistent typing signal. It is above noise in every seed (0.101, 0.118, 0.108 vs about 0.067). With audio + visual it reaches 0.146 vs 0.062.
- Typing → response is modestly above noise: keypress 0.137 vs 0.107 (3 seeds), and continuous + audio + visual 0.140 vs 0.092.
- The word-level format is at noise level so far (0.052 vs 0.049 at 90% masking). One epoch per word gives little EEG per query (median 6 units).
- These results are not directly comparable to typing v1, because the ICT target and the candidate index both changed.

### Training dynamics

- Typing-only runs peaked early. Continuous trained longest (best epoch 9–12); keypress and word peaked at epochs 1–3.
- In the external-data arms, the best typing validation loss came at epoch 1–5 and did not improve over the next 20 epochs. The saved checkpoint is therefore an early model. This is also why Alice and Nieuwland score lower here (about 0.12–0.18) than in earlier runs (about 0.19).

### Next steps

1. Pretrain on Alice + Nieuwland, then fine-tune on typing, so typing starts from a model that has already learned from the external data instead of competing with it during training.
2. Give the word-level format more EEG per word, for example several epochs across the time it took to type the word, not just one at its first key.

## 2026-10-29: Imagined-Query Runs

### Data

- The imagining interval is the 10 s fixation period between reading the question and typing.
- Pairs are built by `Audio_BPR/Dataset/authentic_imagining_reader.py` into `dataset/authentic_imagining_pairs/`. There are 1,055 trials from 25 participants; P003 and P007 are included.
- Main task: imagined EEG → `response_seen`.
- Imagining ICT: two 2–4 s crops per trial (2,110 per variant), each paired with the displayed question with a span masked. Three masking variants:
  - Position-aligned: the words at the same relative position in the question as the crop is in the window.
  - Pace-aligned: the words the participant's mean reading fixation duration places inside the crop. 1,335 crops fall after the question would have been finished and are left unmasked.
  - Random span: a random span of the same length (control).
- Code: `Audio_BPR` commit `47acfcb`. Jobs `8241979`–`8241992`; results in `Audio_BPR/prepared_experiments_20261028_imagining_v2{,_256G}/`.

### Runs

- Method comparison (imagined EEG → response, imagining data only):
  - Window: all ten 1 s units, mean-pooled (baseline).
  - Crop: random 2–4 s crops during training; full window at test.
  - Crop-average: as Crop, but tested on the average of 5 evenly spaced 3 s crops.
  - Sliding window: 2 s windows with a 1 s step; each trial is scored by its best-matching window.
  - Transfer: starts from the reading arm A model.
  - Pace-aligned / Uniform / Shuffled: one segment per typed word, each the length of that participant's mean fixation duration, placed end to end from the start / spread evenly / at random positions.
- Arms:
  - A: imagined → response plus imagining ICT, once for each masking variant.
  - B: A (position-aligned) + Alice.
  - C: A (position-aligned) + Nieuwland.
  - D: A (position-aligned) + Alice + Nieuwland (256 GB).
- Settings: learning rate 3e-5, patience 20, no early stopping before epoch 10, gradient clip 5, early stopping on imagining validation loss, original Audio BPR split, seed 42. B–D use source balancing.

### Full pooled test index, MRR

The method runs are not masked, so their score is the same at every masking level.

| Arm | 0% | 25% | 50% | 75% | 90% | 100% | Noise | Best epoch |
|---|---|---|---|---|---|---|---|---|
| Window (baseline) | 0.108 | 0.108 | 0.108 | 0.108 | 0.108 | 0.108 | 0.115 | 2 |
| Crop | 0.094 | 0.094 | 0.094 | 0.094 | 0.094 | 0.094 | 0.088 | 21 |
| Crop-average | 0.108 | 0.108 | 0.108 | 0.108 | 0.108 | 0.108 | 0.087 | 21 |
| Sliding window | 0.091 | 0.091 | 0.091 | 0.091 | 0.091 | 0.091 | 0.097 | 1 |
| Transfer | 0.111 | 0.111 | 0.111 | 0.111 | 0.111 | 0.111 | 0.112 | 2 |
| Pace-aligned | 0.104 | 0.104 | 0.104 | 0.104 | 0.104 | 0.104 | 0.105 | 1 |
| Uniform | 0.101 | 0.101 | 0.101 | 0.101 | 0.101 | 0.101 | 0.113 | 1 |
| Shuffled | 0.075 | 0.075 | 0.075 | 0.075 | 0.075 | 0.075 | 0.119 | 1 |
| A: position-aligned ICT | 0.075 | 0.068 | 0.058 | 0.058 | 0.055 | 0.054 | 0.056 | 3 |
| A: pace-aligned ICT | 0.091 | 0.085 | 0.082 | 0.080 | 0.076 | 0.075 | 0.047 | 14 |
| A: random-span ICT | 0.095 | 0.080 | 0.069 | 0.066 | 0.063 | 0.064 | 0.035 | 14 |
| B: A + audio | 0.408 | 0.276 | 0.230 | 0.209 | 0.205 | 0.212 | 0.008 | 190 |
| C: A + visual | 0.374 | 0.261 | 0.212 | 0.195 | 0.188 | 0.195 | 0.013 | 180 |
| D: A + audio + visual | 0.397 | 0.259 | 0.217 | 0.196 | 0.190 | 0.196 | 0.006 | 100 |

### Full pooled test index, Hit@10

| Arm | 0% | 25% | 50% | 75% | 90% | 100% | Noise |
|---|---|---|---|---|---|---|---|
| Window | 0.274 | 0.274 | 0.274 | 0.274 | 0.274 | 0.274 | 0.292 |
| Crop | 0.208 | 0.208 | 0.208 | 0.208 | 0.208 | 0.208 | 0.255 |
| Crop-average | 0.217 | 0.217 | 0.217 | 0.217 | 0.217 | 0.217 | 0.264 |
| Sliding window | 0.245 | 0.245 | 0.245 | 0.245 | 0.245 | 0.245 | 0.255 |
| Transfer | 0.292 | 0.292 | 0.292 | 0.292 | 0.292 | 0.292 | 0.292 |
| Pace-aligned | 0.189 | 0.189 | 0.189 | 0.189 | 0.189 | 0.189 | 0.245 |
| Uniform | 0.226 | 0.226 | 0.226 | 0.226 | 0.226 | 0.226 | 0.283 |
| Shuffled | 0.170 | 0.170 | 0.170 | 0.170 | 0.170 | 0.170 | 0.226 |
| A: position-aligned ICT | 0.183 | 0.139 | 0.126 | 0.114 | 0.110 | 0.107 | 0.079 |
| A: pace-aligned ICT | 0.211 | 0.196 | 0.202 | 0.170 | 0.174 | 0.170 | 0.104 |
| A: random-span ICT | 0.208 | 0.192 | 0.177 | 0.155 | 0.158 | 0.158 | 0.079 |
| B: A + audio | 0.615 | 0.472 | 0.433 | 0.406 | 0.392 | 0.392 | 0.013 |
| C: A + visual | 0.496 | 0.423 | 0.396 | 0.384 | 0.378 | 0.377 | 0.019 |
| D: A + audio + visual | 0.556 | 0.442 | 0.399 | 0.389 | 0.384 | 0.387 | 0.010 |

### Per source, MRR at 90% masking / noise

The imagined → response pool has about 40 responses (chance about 0.107).

| Arm | Imagined → response | Imagining ICT | Alice | Nieuwland |
|---|---|---|---|---|
| Methods (all 8) | 0.075–0.111 / 0.10–0.12 | — | — | — |
| A: position-aligned | 0.130 / 0.144 | 0.050 / 0.054 | — | — |
| A: pace-aligned | 0.138 / 0.108 | 0.094 / 0.061 | — | — |
| A: random-span | 0.156 / 0.104 | 0.047 / 0.034 | — | — |
| B: + audio | 0.395 / 0.099 | 0.203 / 0.036 | 0.203 / 0.007 | — |
| C: + visual | 0.460 / 0.089 | 0.222 / 0.035 | — | 0.174 / 0.012 |
| D: + audio + visual | 0.347 / 0.090 | 0.186 / 0.025 | 0.204 / 0.013 | 0.181 / 0.010 |

### Findings

- On imagining data alone, the windowing and pacing methods are at chance, and most reach their best validation loss at epoch 1–2. About 850 training pairs pointing at about 48 responses is not enough training signal.
- Arm A with imagining ICT is mixed: the pace-aligned and random-span versions are modestly above noise; the position-aligned version is not.
- Adding external data makes imagining learn. Imagined → response reaches about 4–5× noise:
  - C (+ visual) is best: 0.460 vs 0.089.
  - B (+ audio): 0.395 vs 0.099.
  - D (+ both): 0.347 vs 0.090. It is below either alone, likely because balancing gives imagining a third of the training samples instead of half.
- Imagining ICT also rises to about 0.19–0.22 vs about 0.03.
- Visual reading EEG (Nieuwland) helping imagining most is the opposite of reading and typing, where visual data hurt.
- Alice and Nieuwland themselves are unaffected (about 0.17–0.20).

### Caveats

- B–D differ from A in three ways at once: external data, source balancing (which oversamples imagining pairs), and much longer training (best epoch 100–190 vs 3–14).
- Possible trial overlap. The split assigns individual pairs, not trials, so a trial's ICT crops (slices of its imagining window) can be in training while its full window is in test. Over 100+ epochs the model can learn to recognise that trial's EEG, and the response and question share a topic. The same applies, more mildly, to reading arm A, whose question-ICT spans are subsets of the full-question fixation epochs.
- Single seed; 84–115 imagined → response test queries per arm.

### Next steps

1. Rerun imagining arms A–D (and reading arm A for comparison) with the trial-grouped split (`--split-strategy trial`), so each trial's crops and full window stay in the same split.
2. Separate the external data from the extra training: run arm A with source balancing and a fixed long run (e.g. 150 epochs) but no Alice or Nieuwland.
3. Run 3 seeds for arms B and C, ideally under the trial-grouped split.
4. Find out why visual helps: pretrain on Nieuwland only and fine-tune on imagining, then compare with transfer from reading arm A.
5. Re-run the windowing methods inside arm C (crop, crop-average, sliding window, pace / uniform / shuffled), now that the model learns.
6. Use the sliding-window scores to find when, within the 10 s window, the query content appears.
7. Combined targets: treat the question and the response both as correct for the full window, using the best arm.
8. Compare reading, typing and imagining on the same trials with the trial-grouped split.
9. Housekeeping: storage is over quota (829 GB vs 350 GB). The unused question and typed-query imagining files (about 8 GB) can be deleted.

Steps 1–3 come first: they are cheap and decide whether the imagining result holds before investing in steps 4–8.

## 2026-10-30: Imagining v3 (Trial-Grouped Split) — First Results

### Why these runs were needed

All earlier runs used the original Audio BPR split, which assigns individual pairs to train, validation and test at random. Several pairs come from the same trial's EEG, so one trial could be split across training and test:

- Imagining: a trial's ICT crops (2–4 s slices of its imagining window) could be in training while that trial's full 10 s window was in test. The test query then contained EEG the model had already been trained on.
- Reading: the question-ICT spans are subsets of the same fixation epochs used for the full-question query, so the same overlap applied.

With long training, the model can learn to recognise a specific trial's EEG rather than anything general. Because the question and the response share a topic, that is enough to score well at test. The trial-grouped split (`--split-strategy trial`, split seed 42) keeps all of a trial's pairs in the same split. A check confirmed no trial is shared between train and test, or between train and validation (2,547 / 291 / 327 imagining pairs).

### Purpose of each completed run

| Run | Question it answers |
|---|---|
| Imagining A (position, pace and random-span ICT) | Does imagining still work when each trial's data stays entirely in training or entirely in test? |
| Imagining A, balanced, fixed 150 epochs | Does longer, balanced training help without external data? |
| Imagining A, transfer from reading | Does starting from a reading-trained model help? |
| Reading arm A, trial-grouped split | Does reading's earlier good result survive the stricter split? |

### Results (MRR at 90% masking; chance about 0.10)

The main task is imagined EEG → response (imagining) or question-reading EEG → response (reading).

| Run | Main task | Noise | ICT | Noise |
|---|---|---|---|---|
| Imagining A, position-aligned ICT | 0.102 | 0.091 | 0.041 | 0.035 |
| Imagining A, pace-aligned ICT | 0.097 | 0.086 | 0.048 | 0.058 |
| Imagining A, random-span ICT | 0.095 | 0.084 | 0.047 | 0.036 |
| Imagining A, balanced, 150 epochs | 0.178 | 0.092 | 0.071 | 0.023 |
| Imagining A, transfer from reading | 0.104 | 0.087 | 0.039 | 0.035 |
| Reading arm A, trial-grouped split | 0.116 | 0.102 | 0.032 | 0.043 |

Reading arm A under the two splits:

| Split | Query → response | Noise |
|---|---|---|
| Original (pair-level) | 0.293 | 0.129 |
| Trial-grouped (stricter) | 0.116 | 0.102 |

### What this tells us so far

- The stricter split removes most of the earlier gains. Reading arm A drops from well above noise to close to noise, so much of the earlier result likely came from the same trial's EEG appearing in both training and test.
- Long, balanced training is the only imagining run clearly above noise: about 2× noise on the main task and 3× on ICT, without external data.
- The ICT masking choice made no difference: position, pace and random-span masking all score at noise level.
- Transfer from reading did not help; it matched training from scratch.
- The external-data runs (arms B, C, D, the seeds, the Nieuwland transfer, and the methods inside arm C) are still missing; see below.

### What "balanced" means

`--balance_sources` replaces the normal shuffle with weighted sampling. Each source gets an equal share of the draws in an epoch, regardless of its size. Pairs are drawn with replacement, so pairs from smaller sources repeat within an epoch and some from larger sources are skipped. An epoch keeps the same total number of draws.

For imagining arm A (about 2,550 training pairs):

| Source | Training pairs | Draws per epoch, unbalanced | Draws per epoch, balanced |
|---|---|---|---|
| Imagined window → response | ~850 | ~850 (each once) | ~1,275 (each ~1.5×) |
| Imagining ICT crops → question | ~1,700 | ~1,700 (each once) | ~1,275 (each ~0.75×) |

Balancing gives the main task a larger share of training. In arms B–D it also stops Alice and Nieuwland (about 6,500 pairs each) from swamping the imagining data.

The "balanced, 150 epochs" run changed two things compared with plain arm A:

| Setting | Arm A (position-aligned) | Arm A, balanced, 150 epochs |
|---|---|---|
| Sampling | Each pair once per epoch | Balanced |
| Training length | Stops early (patience 20, at least 10 epochs) | Fixed 150 epochs |
| Imagined → response | 0.102 / 0.091 noise | 0.178 / 0.092 noise |

So we cannot yet say whether balancing or longer training caused the improvement. One extra run would separate them: arm A for 150 epochs without balancing.

### Other issues found

- Missing input files: `/users/gxb18167/BPR_mean_pooling_inputs/` (the Alice and Nieuwland pair files) had been deleted, so all 15 runs using external data failed at startup. Both files were regenerated on 2026-10-30 (job `8245155`) from the raw data (`~/Alice`, `~/Processed segmented data`, `~/Sentence Materials`) using the original readers and seed 42. They match the originals: 8,134 Alice and 8,132 Nieuwland pairs, with the same EEG shapes (6 × 85 × 128 and 6 × 250 × 32). The Alice reader's hard-coded old data path was redirected to `~/Alice` on a temporary copy; the reader itself was not changed.
- Checkpoint bug (in the original Audio BPR code): the best model is saved with `model.state_dict().copy()`, which copies only the dictionary, not the weights. The "saved" weights keep changing during training, so every run so far has been tested with the final-epoch model (best epoch plus patience), not the best one. This affects all runs equally, reading and typing included, and explains why every run reports "best epoch 1" yet scores differently. Pending decision: fix it (test the genuinely best epoch) or keep the original behaviour for comparability.

### Next steps

1. Decide on the checkpoint fix, then resubmit the 15 failed imagining runs (arms B–D, seeds, Nieuwland transfer, methods inside arm C, combined targets).
2. Run arm A for 150 epochs without balancing, to separate balancing from training length.
