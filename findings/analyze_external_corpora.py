"""Summarize lexical properties and overlap of task, Alice and Nieuwland ICT texts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "dataset/authentic_task_ict_query_response/authentic_query_response_aligned_pairs.npy"
EXTERNAL = Path.home() / "BPR_mean_pooling_inputs"
SOURCES = {
    "task": TASK,
    "alice": EXTERNAL / "alice_ict_pairs_RUNTIME_MASKING.npy",
    "nieuwland_all_labs": EXTERNAL / "nieuwland_all_labs_ict_pairs_RUNTIME_MASKING.npy",
}
OUT = ROOT / "findings"
TOKEN = re.compile(r"[a-z0-9]+(?:['’][a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


def summarize_source(name: str, path: Path) -> tuple[dict, set[str], set[str]]:
    dataset = np.load(path, allow_pickle=True).item()
    pairs = dataset["ict_pairs"]
    queries, docs, participants = set(), set(), set()
    vocabulary: set[str] = set()
    query_lengths, doc_lengths, coverage, jaccards = [], [], [], []
    for pair in pairs:
        query = tokenize(str(pair.get("query_text", "")))
        doc = tokenize(str(pair.get("doc_text", "")))
        if not query or not doc:
            continue
        query_set, doc_set = set(query), set(doc)
        queries.add(" ".join(query))
        docs.add(" ".join(doc))
        vocabulary.update(doc_set)
        query_lengths.append(len(query))
        doc_lengths.append(len(doc))
        coverage.append(sum(token in doc_set for token in query) / len(query))
        jaccards.append(len(query_set & doc_set) / len(query_set | doc_set))
        participant = pair.get("participant_id")
        if participant is not None:
            participants.add(str(participant))
    summary = {
        "source": name,
        "pair_count": len(pairs),
        "participant_count": len(participants),
        "unique_queries": len(queries),
        "unique_documents": len(docs),
        "document_vocabulary_size": len(vocabulary),
        "mean_query_tokens": float(np.mean(query_lengths)),
        "median_query_tokens": float(np.median(query_lengths)),
        "mean_document_tokens": float(np.mean(doc_lengths)),
        "median_document_tokens": float(np.median(doc_lengths)),
        "mean_query_token_coverage_in_document": float(np.mean(coverage)),
        "mean_query_document_token_jaccard": float(np.mean(jaccards)),
    }
    return summary, vocabulary, docs


def main() -> None:
    summaries, vocabularies, documents = [], {}, {}
    for name, path in SOURCES.items():
        summary, vocabularies[name], documents[name] = summarize_source(name, path)
        summaries.append(summary)
        print(name, json.dumps(summary, sort_keys=True), flush=True)

    OUT.mkdir(exist_ok=True)
    with (OUT / "external_corpus_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "External-corpus lexical summary",
        "================================",
        "Tokens are lowercase alphanumeric word forms; no stemming or semantic embeddings are used.",
        "Query-document coverage is the share of query tokens appearing anywhere in the paired document.",
        "",
    ]
    for summary in summaries:
        lines.append(
            f"{summary['source']}: {summary['pair_count']} pairs, {summary['participant_count']} "
            f"participants, {summary['unique_documents']} unique documents, "
            f"mean document length {summary['mean_document_tokens']:.1f} tokens, "
            f"query-token coverage {summary['mean_query_token_coverage_in_document']:.3f}."
        )
    lines.extend(["", "Cross-source document and vocabulary overlap", "-------------------------------------------"])
    names = list(SOURCES)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared_vocab = vocabularies[left] & vocabularies[right]
            vocab_union = vocabularies[left] | vocabularies[right]
            shared_docs = documents[left] & documents[right]
            lines.append(
                f"{left} vs {right}: {len(shared_docs)} exact shared normalized documents; "
                f"{len(shared_vocab)} shared word forms; vocabulary Jaccard "
                f"{len(shared_vocab) / len(vocab_union):.4f}."
            )
    (OUT / "external_corpus_summary.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()