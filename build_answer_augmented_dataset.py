"""Join participant-visible responses with the curated query answer key."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent / "dataset"
SOURCE_FIELDS = ("question_text", "typed_text", "condition", "chatbot_condition", "answer_text")
OUTPUT_FIELDS = (
    "participant_id", "session_id", "part", "trial", "question_id", "question_text",
    "typed_text", "condition", "chatbot_condition", "response_seen", "ground_truth_answer",
    "ground_truth_response", "answer_status", "answer_provenance", "answer_notes",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def build(force: bool = False) -> int:
    key_rows = read_csv(DATASET_DIR / "query_answer_key.csv")
    reference_rows = read_csv(DATASET_DIR / "query_reference_responses.csv")
    answer_key = {row["question_id"]: row for row in key_rows}
    references = {row["question_id"]: row["ground_truth_response"] for row in reference_rows}
    output_path = DATASET_DIR / "trial_response_ground_truth.csv"
    if output_path.exists() and not force:
        raise FileExistsError(f"{output_path} already exists; rerun with --force to replace it")

    source_paths = sorted(DATASET_DIR.glob("P*/*/*_judgement_stages.csv"))
    if not source_paths:
        raise FileNotFoundError(f"No judgement-stage files found under {DATASET_DIR}")

    trials: OrderedDict[tuple[str, str, str, str, str], dict[str, str]] = OrderedDict()
    for path in source_paths:
        session_id = path.parent.name
        participant_id = path.parents[1].name
        for row in read_csv(path):
            question_id = row.get("question_id", "")
            if question_id not in answer_key or question_id not in references:
                raise ValueError(f"No ground-truth mapping for {question_id!r} in {path}")
            if not row.get("answer_text", "").strip():
                raise ValueError(f"Empty participant-visible response for {question_id!r} in {path}")

            key = (participant_id, session_id, row.get("part", ""), row.get("trial", ""), question_id)
            signature = tuple(row.get(field, "") for field in SOURCE_FIELDS)
            if key in trials:
                previous = tuple(
                    trials[key]["response_seen"] if field == "answer_text" else trials[key][field]
                    for field in SOURCE_FIELDS
                )
                if previous != signature:
                    raise ValueError(f"Judgement stages disagree on trial data for {key}")
                continue

            source_answer = answer_key[question_id]
            if row.get("question_text", "") != source_answer["question_text"]:
                raise ValueError(f"Question text differs from answer key for {question_id!r} in {path}")
            trials[key] = {
                "participant_id": participant_id,
                "session_id": session_id,
                "part": row.get("part", ""),
                "trial": row.get("trial", ""),
                "question_id": question_id,
                "question_text": row.get("question_text", ""),
                "typed_text": row.get("typed_text", ""),
                "condition": row.get("condition", ""),
                "chatbot_condition": row.get("chatbot_condition", ""),
                "response_seen": row.get("answer_text", ""),
                "ground_truth_answer": source_answer["ground_truth_answer"],
                "ground_truth_response": references[question_id],
                "answer_status": source_answer["answer_status"],
                "answer_provenance": source_answer["answer_provenance"],
                "answer_notes": source_answer["notes"],
            }

    observed_ids = {row["question_id"] for row in trials.values()}
    if observed_ids != set(answer_key) or observed_ids != set(references):
        raise ValueError(
            f"Query coverage mismatch: missing={sorted(set(answer_key) - observed_ids)}, "
            f"unexpected={sorted(observed_ids - set(answer_key))}"
        )

    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trials.values())
    return len(trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace an existing joined CSV")
    arguments = parser.parse_args()
    count = build(force=arguments.force)
    print(f"Wrote {count} participant/session/trial rows to {DATASET_DIR / 'trial_response_ground_truth.csv'}")