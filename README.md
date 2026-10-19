# Authentic_Query_BPR

## EEG Extraction With Typing Timings

Run the batch extractor with an EEG-capable Python environment (`numpy`,
`pandas`, `mne`, and `pyxdf`) and a local BMI4IRDS checkout:

```bash
python extract_dataset.py --library-dir ~/BMI4IRDS \
	--base-dir "$HOME/Task 2 - Niall" \
	--output-dir ./extracted_with_typing --participants P012
```

The output directory must be new or empty. Existing datasets are not modified.
Use `--dry-run` to inspect input selection before exporting. Existing identity
checks and manifest confirmations still apply.

Full typing-period EEG remains in `participants/<ID>/typing_raw_segments/`,
indexed as `phase=typing` in `segments.csv`. Three additional tables are exported:

- `typing_keystrokes.csv`: logged keys, including deleted characters, modifiers,
	source log line numbers, and validation flags.
- `typing_trials.csv`: participant/part/global trial/question links, submitted
	and reconstructed text, clock checks, validation status, and full EEG file.
- `typing_word_timings.csv`: validated final-query words, first/latest surviving
	character timestamps, and following-space or submission completion timestamps.

The selected PsychoPy CSV's same-basename `.log` is used automatically. Other
logs are never selected arbitrarily. An input manifest can explicitly select a
log using `psychopy_log`, alongside existing `eeg_file`, `psychopy_csv`, and
identity-confirmation fields. Relative manifest paths are relative to `--base-dir`.
Missing logs, failed reconstruction, and annotation errors do not discard the
full typing-period EEG. Coverage counts/errors are recorded in the dataset
manifest, and typing segments include timing/alignment status columns.

### Validation And Clocks

This checkout uses the local standard-library `typing_annotations.py` parser;
the handoff's `DataExtractors/TypingTimingExtractor.py` was not present in the
installed BMI4IRDS copy. Typing windows come from `textbox`, `textbox_2`, and
`textbox_3` autoDraw events. Reconstruction supports character entry, backspace,
delete, left/right, and home/end. Validation ignores case and terminal submission
newlines and allows standard shifted punctuation confirmed by submitted text.
Unsupported keys/shortcuts, selection, paste, layout differences, and multiline
editing are not reliably reconstructed. Such trials are flagged when detectable;
text equality cannot prove that no unlogged editing occurred.

For each trial, the log start must match a unique CSV `textbox*.started` value,
and the log end must match the phase's CSV typing offset, both within 50 ms.
This verifies compatibility with the CSV timebase; no log clock correction is
estimated or silently applied. Unmatched/incomplete windows and text or clock
mismatches retain raw events but produce no word rows.

`log_time_s`, `first_surviving_key_s`, `last_surviving_key_s`, and `completion_s`
are in the original log timebase. EEG coordinates are supplied only for validated
trials with successful marker alignment passing BMI4IRDS's frame residual check.
`*_eeg_time_s` uses the pipeline's EEG clock; `*_eeg_sample` is a zero-based sample
relative to the continuous Raw data (not an MNE absolute event sample), and
`*_segment_sample` is relative to the saved full typing segment. Out-of-recording
indices, unavailable segment offsets, and unverified mappings are blank, not
clamped. Alignment report paths are included for inspection.

Completion is the following space, or submission for the final word. Without a
submission key, the routine end is explicitly labeled `routine_end_proxy`.
If a correction happens after its word's following space, completion is blank
and labeled `edited_after_boundary`. Deleted words remain visible only in raw
keys. These are annotations, not separately cropped word/keypress EEG epochs;
they reference the preserved full typing EEG for downstream window selection.

Focused tests:

```bash
python -m unittest test_extract_dataset -v
```