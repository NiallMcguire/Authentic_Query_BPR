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
