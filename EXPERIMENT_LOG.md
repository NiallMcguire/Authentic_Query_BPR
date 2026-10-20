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
