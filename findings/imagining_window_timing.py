"""When within the 10 s imagining window does the query content appear? (next step 6)

Loads a trained sliding-window (multiple-instance) model, scores every 2 s window of each test
imagined->response query against its correct response, and reports which window matches best.
Usage: python imagining_window_timing.py <prepared_experiments_dir> <arm_name>
"""

from __future__ import annotations

import csv
import shlex
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

AUDIO_BPR = Path(__file__).resolve().parents[1] / "Audio_BPR"
sys.path.insert(0, str(AUDIO_BPR))
from controller import create_dataloaders, handle_multiple_datasets  # noqa: E402
from mv_models import create_model  # noqa: E402


def job_args(prepared: Path, arm: str) -> dict:
    tokens = shlex.split((prepared / "jobs" / f"{arm}.sbatch").read_text().strip().splitlines()[-1])
    def value(flag, default=None, count=1):
        if flag not in tokens:
            return default
        index = tokens.index(flag)
        return tokens[index + 1] if count == 1 else tokens[index + 1:index + 1 + count]
    start = tokens.index("--data_paths") + 1 if "--data_paths" in tokens else tokens.index("--data_path") + 1
    end = next(i for i in range(start, len(tokens)) if tokens[i].startswith("--"))
    return {"paths": tokens[start:end], "split": value("--split-strategy", "original"),
            "split_seed": int(value("--split-seed", value("--seed", "42"))),
            "unit_shape": [int(v) for v in value("--eeg-unit-shape", None, 2) or []] or None,
            "window": int(value("--eeg-window-units", "2")), "stride": int(value("--eeg-window-stride", "1"))}


def main(prepared: Path, arm: str) -> None:
    args = job_args(prepared, arm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combined, dims = handle_multiple_datasets(args["paths"], 50)
    tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
    _, _, test_loader, dims = create_dataloaders(
        None, tokenizer, 32, 256, 50, 0.8, False, 90, dims, combined, args["split_seed"], False,
        args["split"], eeg_unit_shape=args["unit_shape"])
    model = create_model(colbert_model_name="colbert-ir/colbertv2.0", hidden_dim=768, eeg_arch="simple",
                         device=device, use_lora=False, pooling_strategy="mean", encoder_type="dual",
                         global_eeg_dims=dims, query_type="eeg", use_pretrained_text=True)
    model.load_state_dict(torch.load(prepared / "results" / arm / "model_mean_dual.pt",
                                     map_location=device)["model_state_dict"])
    model.to(device).eval()
    model.eeg_window_units, model.eeg_window_stride = args["window"], args["stride"]

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            keep = [i for i, meta in enumerate(batch["metadata"])
                    if str(meta["dataset_source"]).startswith("authentic_imagining_window_response")]
            if not keep:
                continue
            windows = model.encode_eeg(batch["eeg_queries"][keep].to(device), batch["eeg_mv_masks"][keep].to(device))
            docs = model.encode_text(batch["docs"]["input_ids"][keep].to(device),
                                     batch["docs"]["attention_mask"][keep].to(device))
            sims = (F.normalize(windows, dim=-1) * F.normalize(docs, dim=-1)).sum(-1).cpu().numpy()
            for i, row in zip(keep, sims):
                meta = batch["metadata"][i]
                rows.append({"participant_id": meta["participant_id"], "best_window": int(row.argmax()),
                             **{f"window_{w}": float(s) for w, s in enumerate(row)}})

    n_windows = max(int(key.split("_")[1]) for key in rows[0] if key.startswith("window_")) + 1
    out = Path(__file__).resolve().parent / f"imagining_window_timing_{arm}"
    with open(out.with_suffix(".csv"), "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = np.bincount([r["best_window"] for r in rows], minlength=n_windows)
    mean_sim = np.array([[r[f"window_{w}"] for w in range(n_windows)] for r in rows]).mean(0)
    labels = [f"{w * args['stride']}-{w * args['stride'] + args['window']}s" for w in range(n_windows)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].bar(labels, best, color="tab:purple")
    axes[0].set_title(f"Best-matching window per test query (n={len(rows)})")
    axes[0].set_ylabel("Queries")
    axes[1].plot(labels, mean_sim, marker="o", color="tab:purple")
    axes[1].set_title("Mean similarity to the correct response")
    for axis in axes:
        axis.set_xlabel("Window within the 10 s imagining interval")
        axis.tick_params(axis="x", rotation=45)
    fig.suptitle(f"When does imagined query content appear? ({arm})")
    fig.savefig(out.with_suffix(".png"), dpi=160)
    print(f"Saved {out.with_suffix('.png')} and {out.with_suffix('.csv')}")
    print("best-window counts:", dict(zip(labels, best.tolist())))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2])
