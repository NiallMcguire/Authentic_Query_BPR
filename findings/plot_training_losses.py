"""Plot train and primary-validation losses for the Task query-response ablations."""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "Audio_BPR"
BATCHES = (
    {
        "name": "response-only index",
        "log_dir": LOG_ROOT / "prepared_experiments_20261020_query_response_ablations" / "logs",
        "output_stem": "training_validation_losses",
        "runs": (
            ("Query-response only", "query_response_only", "8208840"),
            ("+ Question ICT", "query_response_plus_question_ict", "8208841"),
            ("+ Response ICT", "query_response_plus_response_ict", "8208842"),
            ("All Task sources", "all_task_sources", "8208836"),
            ("All Task + Audio", "all_task_sources_audio", "8208837"),
            ("All Task + Visual", "all_task_sources_visual", "8208839"),
            ("All Task + Audio + Visual", "all_task_sources_audio_visual", "8208838"),
        ),
    },
    {
        "name": "expanded question + response index",
        "log_dir": LOG_ROOT / "prepared_experiments_20261021_expanded_index" / "logs",
        "output_stem": "training_validation_losses_expanded_index",
        "runs": (
            ("Query-response only", "query_response_only", "8209433"),
            ("+ Question ICT", "query_response_plus_question_ict", "8209434"),
            ("+ Response ICT", "query_response_plus_response_ict", "8209435"),
            ("All Task sources", "all_task_sources", "8209429"),
            ("All Task + Audio", "all_task_sources_audio", "8209430"),
            ("All Task + Visual", "all_task_sources_visual", "8209432"),
            ("All Task + Audio + Visual", "all_task_sources_audio_visual", "8209431"),
        ),
    },
)
SUMMARY_PATTERN = re.compile(r"Epoch\s+(\d+)/\d+\s+Summary:")
TRAIN_PATTERN = re.compile(r"Train\s+-\s+Loss:\s*([0-9.eE+-]+)")
VALIDATION_PATTERN = re.compile(r"Val\s+-\s+Loss:\s*([0-9.eE+-]+)")


def read_history(log_dir: Path, condition: str, slug: str, job_id: str) -> list[dict]:
    log_path = log_dir / f"{slug}-{job_id}.out"
    lines = log_path.read_text(errors="replace").splitlines()
    history = []
    for line_index, line in enumerate(lines):
        match = SUMMARY_PATTERN.search(line)
        if not match:
            continue
        summary = " ".join(lines[line_index:line_index + 5])
        train_match = TRAIN_PATTERN.search(summary)
        validation_match = VALIDATION_PATTERN.search(summary)
        if train_match and validation_match:
            history.append({
                "condition": condition,
                "job_id": job_id,
                "epoch": int(match.group(1)),
                "train_loss": float(train_match.group(1)),
                "primary_validation_loss": float(validation_match.group(1)),
            })
    if not history:
        raise ValueError(f"No epoch loss summaries found in {log_path}")
    return history


def plot_batch(batch: dict) -> None:
    runs = batch["runs"]
    histories = [read_history(batch["log_dir"], *run) for run in runs]
    colors = plt.get_cmap("tab10").colors
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    for index, (label, _, _) in enumerate(runs):
        rows = histories[index]
        epochs = [row["epoch"] for row in rows]
        color = colors[index]
        axes[0].plot(epochs, [row["train_loss"] for row in rows], marker="o",
                     markersize=3, linewidth=1.8, color=color, label=label)
        validation_losses = [row["primary_validation_loss"] for row in rows]
        axes[1].plot(epochs, validation_losses, marker="o", markersize=3,
                     linewidth=1.8, color=color, label=label)
        best_index = min(range(len(rows)), key=lambda i: validation_losses[i])
        axes[1].scatter(epochs[best_index], validation_losses[best_index],
                        marker="*", s=110, color=color, edgecolor="black", linewidth=0.5, zorder=3)

    axes[0].set_title("Training loss")
    axes[0].set_ylabel("Mean in-batch contrastive loss")
    axes[1].set_title("Primary validation loss")
    axes[1].set_ylabel("Query-to-displayed-response loss")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle(f"Task query-to-response BPR ablations: {batch['name']}", fontsize=15)

    output = Path(__file__).resolve().parent
    image_path = output / f"{batch['output_stem']}.png"
    figure.savefig(image_path, dpi=180)
    plt.close(figure)

    csv_path = output / f"{batch['output_stem']}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "condition", "job_id", "epoch", "train_loss", "primary_validation_loss",
        ])
        writer.writeheader()
        for rows in histories:
            writer.writerows(rows)

    print(f"Saved {image_path}")
    print(f"Saved {csv_path}")
    print("condition\tbest_epoch\tbest_primary_validation_loss\tfinal_train_loss")
    for rows in histories:
        best_row = min(rows, key=lambda row: row["primary_validation_loss"])
        final_row = rows[-1]
        print("{}\t{}\t{:.4f}\t{:.4f}".format(
            rows[0]["condition"], best_row["epoch"], best_row["primary_validation_loss"],
            final_row["train_loss"],
        ))


def main() -> None:
    for batch in BATCHES:
        print(f"\n=== {batch['name']} ===")
        plot_batch(batch)


if __name__ == "__main__":
    main()