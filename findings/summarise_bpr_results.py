"""Collect the original-split BPR results and write the progress summary and figures.

Outputs:
- findings/BPR_progress_summary.txt
- findings/summary_figures/*.png

Run with: /usr/bin/python3 findings/summarise_bpr_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BPR = ROOT / "Audio_BPR"
OUT_TXT = ROOT / "findings" / "BPR_progress_summary.txt"
FIG_DIR = ROOT / "findings" / "summary_figures"
LEVELS = [0, 25, 50, 75, 90, 100]

READ = "prepared_experiments_20261022_audio_bpr_originalsplit"
READ_MOD = "prepared_experiments_20261023_qr_qict_modalities"
TYPE1 = "prepared_experiments_20261026_typing"
TYPE2 = "prepared_experiments_20261027_typing_v2"
IMAG = "prepared_experiments_20261028_imagining_v2"


# --------------------------------------------------------------------------- loading
def _load_arm(exp: str, arm: str) -> dict:
    folder = BPR / exp / "results" / arm / "test_by_source"
    pooled = json.loads((folder / "all_sources_metrics.json").read_text())
    sources = {
        path.name[: -len("_metrics.json")]: json.loads(path.read_text())
        for path in folder.glob("*_metrics.json")
        if path.name not in ("all_sources_metrics.json", "combined_task_index_metrics.json")
    }
    return {"pooled": pooled, "sources": sources}


class Run:
    def __init__(self, key, label, exp, arms, roles, ext="none"):
        self.key, self.label, self.ext, self.roles = key, label, ext, roles
        arms = arms if isinstance(arms, list) else [arms]
        self.data = [_load_arm(exp, arm) for arm in arms]
        self.n_seeds = len(arms)

    def get(self, role, level, name):
        """Mean and SD (over seeds) of one metric. role is 'pooled' or a role in self.roles."""
        level_key = "noise" if level == "noise" else f"masking_{level}"
        values = []
        for item in self.data:
            if role == "pooled":
                block = item["pooled"]
            else:
                source = self.roles.get(role)
                block = item["sources"].get(source) if source else None
            if block is None or f"test_ranking/{level_key}/{name}" not in block:
                return None
            values.append(float(block[f"test_ranking/{level_key}/{name}"]))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
        return float(np.mean(values)), sd


def reading_roles():
    roles = {"main": "authentic_query_response", "qict": "authentic_question_ict",
             "rict": "authentic_response_ict"}
    roles.update({"alice": "alice", "nieuwland": "nieuwland"})
    return roles


def typing_roles(fmt, version):
    ict = f"authentic_typing_{fmt}_ict" if version == 1 else f"authentic_typing_{fmt}_question_ict"
    return {"main": f"authentic_typing_{fmt}_query_response", "ict": ict,
            "alice": "alice", "nieuwland": "nieuwland"}


def imagining_roles(fmt="window", ict="position"):
    return {"main": f"authentic_imagining_{fmt}_response", "ict": f"authentic_imagining_ict_{ict}",
            "alice": "alice", "nieuwland": "nieuwland"}


def build_runs():
    r = reading_roles()
    reading = [
        Run("qr", "Query->response only", READ, "query_response_only", r),
        Run("A", "A: QR + question ICT", READ, "query_response_plus_question_ict", r),
        Run("qr_rict", "QR + response ICT", READ, "query_response_plus_response_ict", r),
        Run("all", "QR + question ICT + response ICT", READ, "all_task_sources", r),
        Run("B", "B: A + audio", READ_MOD, "query_response_question_ict_audio", r, "audio"),
        Run("C", "C: A + visual", READ_MOD, "query_response_question_ict_visual", r, "visual"),
        Run("D", "D: A + audio + visual", READ_MOD + "_256G", "query_response_question_ict_audio_visual", r,
            "audio+visual"),
        Run("all_audio", "All task + audio", READ, "all_task_sources_audio", r, "audio"),
        Run("all_visual", "All task + visual", READ, "all_task_sources_visual", r, "visual"),
    ]
    typing_v1 = []
    for fmt in ("continuous", "keypress"):
        roles = typing_roles(fmt, 1)
        name = fmt.capitalize()
        typing_v1 += [
            Run(f"{fmt}", name, TYPE1, f"typing_{fmt}", roles),
            Run(f"{fmt}_audio", f"{name} + audio", TYPE1, f"typing_{fmt}_audio", roles, "audio"),
            Run(f"{fmt}_visual", f"{name} + visual", TYPE1, f"typing_{fmt}_visual", roles, "visual"),
            Run(f"{fmt}_av", f"{name} + audio + visual", TYPE1 + "_256G", f"typing_{fmt}_audio_visual", roles,
                "audio+visual"),
        ]
    typing_v2 = []
    for fmt in ("continuous", "keypress", "word"):
        roles = typing_roles(fmt, 2)
        name = fmt.capitalize() if fmt != "word" else "Word"
        typing_v2 += [
            Run(f"{fmt}", f"{name} (3 seeds)", TYPE2, [f"typing_v2_{fmt}_seed{s}" for s in (42, 43, 44)], roles),
            Run(f"{fmt}_audio", f"{name} + audio", TYPE2, f"typing_v2_{fmt}_audio", roles, "audio"),
            Run(f"{fmt}_visual", f"{name} + visual", TYPE2, f"typing_v2_{fmt}_visual", roles, "visual"),
            Run(f"{fmt}_av", f"{name} + audio + visual", TYPE2 + "_256G", f"typing_v2_{fmt}_audio_visual", roles,
                "audio+visual"),
        ]
    imagining_methods = [
        Run("window", "Window (baseline)", IMAG, "imagining_window_response", imagining_roles()),
        Run("crop", "Crop", IMAG, "imagining_crop_response", imagining_roles()),
        Run("cropavg", "Crop-average", IMAG, "imagining_cropavg_response", imagining_roles()),
        Run("mil", "Sliding window", IMAG, "imagining_mil_response", imagining_roles()),
        Run("transfer", "Transfer from reading A", IMAG, "imagining_transfer_response", imagining_roles()),
        Run("pace", "Pace-aligned segments", IMAG, "imagining_pace_response", imagining_roles("pace")),
        Run("uniform", "Uniform segments", IMAG, "imagining_uniform_response", imagining_roles("uniform")),
        Run("shuffled", "Shuffled segments", IMAG, "imagining_shuffled_response", imagining_roles("shuffled")),
    ]
    imagining_arms = [
        Run("A_position", "A: + position-aligned ICT", IMAG, "imagining_arm_a_ict_position",
            imagining_roles(ict="position")),
        Run("A_pace", "A: + pace-aligned ICT", IMAG, "imagining_arm_a_ict_pace", imagining_roles(ict="pace")),
        Run("A_random", "A: + random-span ICT", IMAG, "imagining_arm_a_ict_random", imagining_roles(ict="random")),
        Run("B", "B: A (position) + audio", IMAG, "imagining_arm_b_audio", imagining_roles(), "audio"),
        Run("C", "C: A (position) + visual", IMAG, "imagining_arm_c_visual", imagining_roles(), "visual"),
        Run("D", "D: A (position) + audio + visual", IMAG + "_256G", "imagining_arm_d_audio_visual",
            imagining_roles(), "audio+visual"),
    ]
    return reading, typing_v1, typing_v2, imagining_methods, imagining_arms


# --------------------------------------------------------------------------- formatting
def fmt_val(value, digits=3):
    if value is None:
        return "-"
    mean, sd = value
    text = f"{mean:.{digits}f}"
    if sd is not None:
        text += f"+/-{sd:.{digits}f}".replace("0.", ".")
    return text


def fmt_int(value):
    return "-" if value is None else f"{value[0]:.0f}"


def pct_over(eeg, noise):
    if eeg is None or noise is None or noise[0] == 0:
        return "-"
    return f"{(eeg[0] - noise[0]) / noise[0] * 100:+.0f}%"


def table(headers, rows, title=None, note=None, n_left=1):
    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
    line = "-" * (sum(widths) + 3 * (len(widths) - 1))

    def render(cells):
        return "   ".join(str(c).ljust(w) if i < n_left else str(c).rjust(w)
                           for i, (c, w) in enumerate(zip(cells, widths)))

    parts = []
    if title:
        parts.append(title)
    parts += [line, render(headers), line, *(render(row) for row in rows), line]
    if note:
        parts.append(note)
    return "\n".join(parts) + "\n"


def pooled_at_90_table(runs, title):
    headers = ["Arm", "#Queries", "#Docs", "MRR", "H@1", "H@5", "H@10", "Noise MRR", "Noise H@10", "MRR vs noise"]
    rows = []
    for run in runs:
        mrr, noise = run.get("pooled", 90, "rr"), run.get("pooled", "noise", "rr")
        rows.append([run.label, fmt_int(run.get("pooled", 90, "num_queries_evaluated")),
                     fmt_int(run.get("pooled", 90, "num_unique_documents")), fmt_val(mrr),
                     fmt_val(run.get("pooled", 90, "hit_at_1")), fmt_val(run.get("pooled", 90, "hit_at_5")),
                     fmt_val(run.get("pooled", 90, "hit_at_10")), fmt_val(noise),
                     fmt_val(run.get("pooled", "noise", "hit_at_10")), pct_over(mrr, noise)])
    return table(headers, rows, title)


def pooled_by_level_table(runs, title, metric="rr", with_docs=True):
    headers = ["Arm"] + [f"{lv}%" for lv in LEVELS] + ["Noise"]
    rows = []
    for run in runs:
        row = [run.label]
        for lv in LEVELS:
            cell = fmt_val(run.get("pooled", lv, metric))
            if with_docs:
                cell += f" [{fmt_int(run.get('pooled', lv, 'num_unique_documents'))}]"
            row.append(cell)
        row.append(fmt_val(run.get("pooled", "noise", metric)))
        rows.append(row)
    note = "  [n] = number of unique documents in the test index at that masking level." if with_docs else None
    return table(headers, rows, title, note)


def per_source_cell(run, role, level=90):
    eeg = run.get(role, level, "rr")
    if eeg is None:
        return "-"
    noise = run.get(role, "noise", "rr")
    docs = run.get(role, level, "num_unique_documents")
    return f"{fmt_val(eeg)} / {fmt_val(noise)} [{fmt_int(docs)}]"


def per_source_table(runs, columns, title):
    headers = ["Arm"] + [name for name, _ in columns]
    rows = [[run.label] + [per_source_cell(run, role) for _, role in columns] for run in runs]
    note = ("  Cells: EEG MRR / noise MRR [#docs]. Each source's test queries are ranked against that\n"
            "  source's own test documents, at 90% masking.")
    return table(headers, rows, title, note)


def method_table(runs, title):
    headers = ["Method", "#Queries", "#Docs", "MRR", "H@1", "H@5", "H@10", "Noise MRR", "Noise H@10"]
    rows = []
    for run in runs:
        rows.append([run.label, fmt_int(run.get("main", 90, "num_queries_evaluated")),
                     fmt_int(run.get("main", 90, "num_unique_documents")), fmt_val(run.get("main", 90, "rr")),
                     fmt_val(run.get("main", 90, "hit_at_1")), fmt_val(run.get("main", 90, "hit_at_5")),
                     fmt_val(run.get("main", 90, "hit_at_10")), fmt_val(run.get("main", "noise", "rr")),
                     fmt_val(run.get("main", "noise", "hit_at_10"))])
    return table(headers, rows, title)


def external_table(groups, title, main_name, ict_role="ict"):
    """groups: list of (group label, [base, +audio, +visual, +both])."""
    headers = ["Group", "Training data", f"{main_name} MRR", "Noise", "vs noise", "vs task only",
               "Task ICT MRR", "Noise", "vs task only"]
    rows = []
    for group, runs in groups:
        base_main, base_ict = runs[0].get("main", 90, "rr"), runs[0].get(ict_role, 90, "rr")
        for run in runs:
            main, ict = run.get("main", 90, "rr"), run.get(ict_role, 90, "rr")
            label = {"none": "Task only", "audio": "+ audio (Alice)", "visual": "+ visual (Nieuwland)",
                     "audio+visual": "+ audio + visual"}[run.ext]
            rows.append([group, label, fmt_val(main), fmt_val(run.get("main", "noise", "rr")),
                         pct_over(main, run.get("main", "noise", "rr")),
                         "-" if run is runs[0] else pct_over(main, base_main), fmt_val(ict),
                         fmt_val(run.get(ict_role, "noise", "rr")),
                         "-" if run is runs[0] else pct_over(ict, base_ict)])
            group = ""
    note = "  Per-source MRR at 90% masking; each source is ranked against its own test documents."
    return table(headers, rows, title, note, n_left=2)


# --------------------------------------------------------------------------- figures
COLOURS = {"none": "#4C72B0", "audio": "#DD8452", "visual": "#55A868", "audio+visual": "#8172B3"}
NOISE_COLOUR = "#C8C8C8"
EXT_LABEL = {"none": "Task only", "audio": "+Audio", "visual": "+Visual", "audio+visual": "+Audio+Visual"}


def _style():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
        "grid.linestyle": ":", "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    })


def _paired_bars(ax, runs, role, labels=None):
    x = np.arange(len(runs))
    eeg = [run.get(role, 90, "rr") for run in runs]
    noise = [run.get(role, "noise", "rr") for run in runs]
    width = 0.38
    for i, (value, run) in enumerate(zip(eeg, runs)):
        if value is None:
            continue
        ax.bar(i - width / 2, value[0], width, color=COLOURS[run.ext], yerr=value[1], capsize=2,
               edgecolor="black", linewidth=0.4)
        ax.text(i - width / 2, value[0] + (value[1] or 0) + 0.005, f"{value[0]:.2f}", ha="center", fontsize=6.5)
    for i, value in enumerate(noise):
        if value is not None:
            ax.bar(i + width / 2, value[0], width, color=NOISE_COLOUR, hatch="///", edgecolor="grey", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels or [EXT_LABEL[r.ext] for r in runs], rotation=30, ha="right")


def fig_external_effect(reading, typing_v2, imagining_arms):
    by = lambda runs, keys: [next(r for r in runs if r.key == k) for k in keys]  # noqa: E731
    panels = [
        ("Reading", by(reading, ["A", "B", "C", "D"]), "Query EEG -> response", "Question ICT", "main", "qict"),
        ("Typing: continuous", by(typing_v2, ["continuous", "continuous_audio", "continuous_visual", "continuous_av"]),
         "Typing EEG -> response", "Typing question ICT", "main", "ict"),
        ("Typing: keypress", by(typing_v2, ["keypress", "keypress_audio", "keypress_visual", "keypress_av"]),
         "Typing EEG -> response", "Typing question ICT", "main", "ict"),
        ("Typing: word", by(typing_v2, ["word", "word_audio", "word_visual", "word_av"]),
         "Typing EEG -> response", "Typing question ICT", "main", "ict"),
        ("Imagining", by(imagining_arms, ["A_position", "B", "C", "D"]),
         "Imagined EEG -> response", "Imagining ICT", "main", "ict"),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(13, 5.6), sharey="row")
    for col, (title, runs, main_name, ict_name, main_role, ict_role) in enumerate(panels):
        _paired_bars(axes[0, col], runs, main_role)
        _paired_bars(axes[1, col], runs, ict_role)
        axes[0, col].set_title(f"{title}\n{main_name}")
        axes[1, col].set_title(ict_name)
        axes[0, col].set_xticklabels([])
    axes[0, 0].set_ylabel("MRR (90% masking)")
    axes[1, 0].set_ylabel("MRR (90% masking)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[k]) for k in COLOURS]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=NOISE_COLOUR, hatch="///", edgecolor="grey"))
    fig.legend(handles, [EXT_LABEL[k] for k in COLOURS] + ["Noise baseline"], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Effect of external data on each modality (per-source index)", y=1.08, fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "fig1_external_data_effect.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_masking_curves(reading, typing_v2, imagining_arms):
    by = lambda runs, keys: [next(r for r in runs if r.key == k) for k in keys]  # noqa: E731
    panels = [
        ("Reading", by(reading, ["qr", "A", "all", "B", "C", "D"])),
        ("Typing (v2)", by(typing_v2, ["continuous", "keypress", "word", "continuous_av", "keypress_av", "word_av"])),
        ("Imagining", by(imagining_arms, ["A_position", "A_pace", "A_random", "B", "C", "D"])),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    markers = ["o", "s", "^", "D", "v", "P"]
    for col, (title, runs) in enumerate(panels):
        cmap = plt.get_cmap("tab10")
        for i, run in enumerate(runs):
            mrr = [run.get("pooled", lv, "rr")[0] for lv in LEVELS]
            docs = [run.get("pooled", lv, "num_unique_documents")[0] for lv in LEVELS]
            noise = run.get("pooled", "noise", "rr")[0]
            colour = cmap(i)
            axes[0, col].plot(LEVELS, mrr, marker=markers[i], color=colour, label=run.label, linewidth=1.3,
                              markersize=4)
            axes[0, col].plot([90], [noise], marker="x", color=colour, markersize=6, linestyle="none")
            axes[1, col].plot(LEVELS, docs, marker=markers[i], color=colour, linewidth=1.0, markersize=3.5)
        axes[0, col].set_title(title)
        axes[0, col].legend(fontsize=6.5, loc="upper right")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Masking level (%)")
        axes[1, col].set_xticks(LEVELS)
    axes[0, 0].set_ylabel("MRR (full pooled index)")
    axes[1, 0].set_ylabel("#Docs in index (log)")
    fig.text(0.5, -0.02, "x = noise baseline (90% masking) in the same colour as its arm", ha="center", fontsize=8)
    fig.suptitle("MRR and index size across masking levels", fontsize=11)
    fig.tight_layout()
    path = FIG_DIR / "fig2_mrr_and_docs_vs_masking.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_imagining_methods(methods, arms):
    runs = methods + arms
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    y = np.arange(len(runs))[::-1]
    h = 0.38
    for yi, run in zip(y, runs):
        eeg, noise = run.get("main", 90, "rr"), run.get("main", "noise", "rr")
        colour = COLOURS[run.ext] if run.ext != "none" else ("#4C72B0" if run in methods else "#6A9BD1")
        ax.barh(yi + h / 2, eeg[0], h, color=colour, edgecolor="black", linewidth=0.4)
        ax.barh(yi - h / 2, noise[0], h, color=NOISE_COLOUR, hatch="///", edgecolor="grey", linewidth=0.4)
        ax.text(eeg[0] + 0.005, yi + h / 2, f"{eeg[0]:.3f}", va="center", fontsize=7)
    ax.axhline(y[len(methods) - 1] - 0.5, color="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r.label for r in runs])
    ax.set_xlabel("Imagined EEG -> response MRR (per-source index, ~40 docs)")
    ax.text(0.98, 0.97, "Imagining data only", transform=ax.transAxes, ha="right", va="top", fontsize=8)
    ax.text(0.98, 0.40, "With imagining ICT / external data", transform=ax.transAxes, ha="right", va="top",
            fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#4C72B0"),
               plt.Rectangle((0, 0), 1, 1, facecolor=NOISE_COLOUR, hatch="///", edgecolor="grey")]
    ax.legend(handles, ["EEG", "Noise baseline"], loc="upper right", bbox_to_anchor=(1.0, 0.9))
    ax.set_title("Imagining: methods and training arms")
    fig.tight_layout()
    path = FIG_DIR / "fig3_imagining_methods.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_typing_formats(typing_v1, typing_v2):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    v2 = [r for r in typing_v2 if r.ext == "none"]
    v1 = [r for r in typing_v1 if r.ext == "none"]
    labels = ["Continuous", "Keypress", "Word"]
    for ax, role, title in [(axes[0], "pooled", "Full pooled index"), (axes[1], "main", "Typing EEG -> response"),
                            (axes[2], "ict", "Typing ICT (v2: displayed question)")]:
        x = np.arange(len(v2))
        w = 0.38
        for i, run in enumerate(v2):
            eeg, noise = run.get(role, 90, "rr"), run.get(role, "noise", "rr")
            ax.bar(i - w / 2, eeg[0], w, yerr=eeg[1], capsize=3, color="#4C72B0", edgecolor="black", linewidth=0.4)
            ax.bar(i + w / 2, noise[0], w, yerr=noise[1], capsize=3, color=NOISE_COLOUR, hatch="///",
                   edgecolor="grey", linewidth=0.4)
            ax.text(i - w / 2, eeg[0] + (eeg[1] or 0) + 0.004, f"{eeg[0]:.3f}", ha="center", fontsize=7)
        if role == "pooled":
            for i, run in enumerate(v1):
                value = run.get(role, 90, "rr")
                ax.plot(i - w / 2, value[0], marker="D", color="#C44E52", markersize=5, linestyle="none",
                        label="Typing v1" if i == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
    axes[0].set_ylabel("MRR (90% masking)")
    handles = [plt.Rectangle((0, 0), 1, 1, color="#4C72B0"),
               plt.Rectangle((0, 0), 1, 1, facecolor=NOISE_COLOUR, hatch="///", edgecolor="grey"),
               plt.Line2D([], [], marker="D", color="#C44E52", linestyle="none")]
    fig.legend(handles, ["Typing v2 EEG (mean +/- SD, 3 seeds)", "Noise baseline", "Typing v1 EEG (seed 42)"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    path = FIG_DIR / "fig4_typing_formats.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_reading_ablation(reading):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=False)
    runs = reading
    for ax, role, title in [(axes[0], "pooled", "Full pooled index"),
                            (axes[1], "main", "Query EEG -> response (per-source index)")]:
        x = np.arange(len(runs))
        w = 0.38
        for i, run in enumerate(runs):
            eeg, noise = run.get(role, 90, "rr"), run.get(role, "noise", "rr")
            ax.bar(i - w / 2, eeg[0], w, color=COLOURS[run.ext], edgecolor="black", linewidth=0.4)
            ax.bar(i + w / 2, noise[0], w, color=NOISE_COLOUR, hatch="///", edgecolor="grey", linewidth=0.4)
            ax.text(i - w / 2, eeg[0] + 0.004, f"{eeg[0]:.2f}", ha="center", fontsize=6.5)
            if role == "pooled":
                docs = run.get(role, 90, "num_unique_documents")[0]
                ax.text(i, -0.035, f"{docs:.0f}", ha="center", fontsize=6.5, color="dimgray")
        ax.set_xticks(x)
        ax.set_xticklabels([r.label for r in runs], rotation=35, ha="right", fontsize=7)
        ax.set_title(title)
        ax.set_ylabel("MRR (90% masking)")
    axes[0].set_ylim(bottom=-0.05)
    axes[0].set_yticks([t for t in axes[0].get_yticks() if 0 <= t <= axes[0].get_ylim()[1]])
    axes[0].text(-0.9, -0.035, "#Docs", fontsize=6.5, color="dimgray", ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[k]) for k in COLOURS]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=NOISE_COLOUR, hatch="///", edgecolor="grey"))
    fig.legend(handles, [EXT_LABEL[k] for k in COLOURS] + ["Noise baseline"], loc="upper center", ncol=5,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    path = FIG_DIR / "fig5_reading_ablations.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- text
HEADER = """\
AUTHENTIC QUERY BPR: PROGRESS SUMMARY
=====================================

Scope: all runs below use the original Audio BPR setup (random pair-level split, seed 42).
Figures for this summary are in findings/summary_figures/.

Terms used throughout
- MRR: mean reciprocal rank of the correct document. H@k: Hit@k.
- Masking level: the share of the ICT span removed from the document text at test time
  (0, 25, 50, 75, 90, 100%). Main-task documents (responses) are never masked.
- #Docs: unique documents in the test index. Documents are de-duplicated by their masked
  text, so the index grows as masking rises.
- Full pooled index: every test query (all sources in the arm) ranked against every
  indexed test document.
- Per-source index: each source's test queries ranked against that source's own test
  documents only. This is the fair way to compare a task across arms.
- Noise baseline: the same test, with the EEG replaced by Gaussian noise (90% masking).
- Audio = Alice (listening EEG, 8,134 ICT pairs). Visual = Nieuwland (reading EEG,
  8,132 ICT pairs). Both come from the original Audio BPR project.
"""

METHODS = """\
1. HOW EACH MODALITY WAS EXTRACTED
==================================

Common to all modalities
- Each EEG "unit" is 513 samples (1 s at 512 Hz). A query is a sequence of units.
- Only participants with verified word-level EEG alignment are used for reading.
- Every pair has a query (EEG) and a positive document (text).

1.1 Reading (question-reading EEG)
- One epoch per eye fixation on a question word, from -200 ms to +800 ms around
  fixation onset.
- Query -> response: all fixation epochs for the whole question -> the response the
  participant was shown for that trial (691 pairs, 25 participants).
- Question ICT: the fixation epochs for a random span of question words -> the full
  question, with that span masked at the training/test masking level (1,328 pairs).
- Response ICT: the fixation epochs for a span of the displayed response -> the full
  displayed response (1,860 pairs).

1.2 Typing (EEG while the participant typed their query)
- 882 validated typing trials from 23 participants (two participants had no keystroke log).
- Keystrokes are matched to the EEG clock, and only keys that survive into the submitted
  query are used.
- Three EEG formats were built from the same trials:
  - Continuous: consecutive 1 s windows from 200 ms before the first key to 800 ms after
    the last key (full sentence as a continuous recording).
  - Keypress: one -200 to +800 ms epoch per surviving keypress.
  - Word: one -200 to +800 ms epoch per typed word, anchored at the word's first key
    (added in typing v2).
- Typing -> response: the full typing EEG -> the response shown for that trial.
- Typing ICT:
  - v1: EEG for a typed span -> the participant's own typed query, span masked
    (1,676 pairs).
  - v2: EEG for a typed span -> the displayed question, span masked. Only spans typed
    verbatim in the question are used (789 pairs). This gives targets shared across
    participants.

1.3 Imagining (EEG while the participant thinks of their query)
- The 10 s fixation period between reading the question and typing the query.
- 1,055 trials from 25 participants.
- Main task: imagined EEG -> the response shown for that trial (never masked).
- EEG formats for the main task:
  - Window: ten consecutive 1 s units covering the whole interval (baseline).
  - Pace-aligned: one segment per typed word, each as long as the participant's mean
    reading fixation (0.19-1.15 s, median 0.33 s), placed end to end from the start.
  - Uniform: the same segments spread evenly across the 10 s (control).
  - Shuffled: the same segments at random positions (control).
- Imagining ICT: two 2-4 s crops per trial (2,110 pairs per variant) -> the displayed
  question with a span masked. Three ways to choose the masked span:
  - Position-aligned: the words at the same relative position in the question as the
    crop is in the 10 s window.
  - Pace-aligned: the words the participant's reading pace places inside the crop.
  - Random span: a random span of the same length (control).


2. HOW WE TRAINED
=================

Shared model and protocol (Audio BPR setup)
- Dual encoder. EEG encoder: simple MLP per unit (hidden size 768), mean pooling over
  units. Text encoder: pretrained ColBERT-v2, frozen.
- Multi-positive in-batch InfoNCE loss: pairs from the same query that share a document
  are not treated as negatives.
- Batch size 32, AdamW, up to 200 epochs, seed 42, 90% masking during training.
- Split: all pairs in an arm are pooled and shuffled (seed 42); 80% train, 20% validation;
  the test set is the second half of validation.
- Test at 0/25/50/75/90/100% masking, plus the noise baseline.

Training settings per modality

  Setting               Reading         Typing v1       Typing v2             Imagining
  --------------------  --------------  --------------  --------------------  ---------------------
  Learning rate         1e-5            1e-5            3e-5                  3e-5
  Patience              5               5               20 (min. 10 epochs)   20 (min. 10 epochs)
  Gradient clip         1.0             1.0             5.0                   5.0
  Max units per query   50              50              100                   (10 units)
  Early stopping on     all sources     all sources     typing sources        imagining sources
  Source balancing      no              no              with external data    with external data
  Seeds                 42              42              42, 43, 44 (task)     42

ICT (inverse cloze task) used in each modality
- Reading: question ICT (span of question -> masked question) and response ICT (span of
  response -> full response).
- Typing v1: typed span -> masked typed query.
- Typing v2: typed span -> masked displayed question.
- Imagining: 2-4 s crop -> masked displayed question (position, pace or random span).

Imagining methods tried on imagined EEG -> response (imagining data only)
- Window: all ten 1 s units, mean-pooled (baseline).
- Crop: random 2-4 s crops during training; full window at test.
- Crop-average: as Crop, but tested on the average of five evenly spaced 3 s crops.
- Sliding window: 2 s windows with a 1 s step; a trial is scored by its best window.
- Transfer: starts from the trained reading arm A model.
- Pace-aligned / Uniform / Shuffled segments (see 1.3).

External data arms
- Each modality was trained on its own task data (arm A), then with Alice added (B),
  Nieuwland added (C) and both added (D).
"""

EXTERNAL_TEXT = """\
3. DID EXTERNAL DATA HELP?
==========================

Short answer
- Pooled-index scores rise sharply with external data in every modality, because the
  external sources are large and learn well. The per-source tables below isolate the
  effect on each task.
- Reading: the task-only arm A gives the strongest query -> response result. Adding
  audio keeps query -> response above noise; visual data gives less.
- Typing: external data gives the best typing -> response results for continuous typing
  (audio + visual: {t_cont_av}) and the best typing question ICT for keypress
  (audio + visual: {t_key_av_ict}).
- Imagining: external data gives the biggest gain of all modalities. Imagined -> response
  rises from {i_a} (task only) to {i_c} with visual data and {i_b} with audio.
  Imagining ICT rises from {i_a_ict} to about {i_bcd_ict}.
- Alice and Nieuwland themselves score about 0.11-0.20 MRR in every mixed arm (noise
  about 0.01).

"""


def build_text(reading, typing_v1, typing_v2, imagining_methods, imagining_arms):
    find = lambda runs, k: next(r for r in runs if r.key == k)  # noqa: E731
    fv = lambda run, role: fmt_val(run.get(role, 90, "rr"))  # noqa: E731
    parts = [HEADER, METHODS]

    ext = EXTERNAL_TEXT.format(
        t_cont_av=fv(find(typing_v2, "continuous_av"), "main"),
        t_key_av_ict=fv(find(typing_v2, "keypress_av"), "ict"),
        i_a=fv(find(imagining_arms, "A_position"), "main"), i_b=fv(find(imagining_arms, "B"), "main"),
        i_c=fv(find(imagining_arms, "C"), "main"),
        i_a_ict=fv(find(imagining_arms, "A_position"), "ict"),
        i_bcd_ict="{:.2f}".format(np.mean([find(imagining_arms, k).get("ict", 90, "rr")[0] for k in "BCD"])),
    )
    parts.append(ext)
    reading_ext = [find(reading, k) for k in ("A", "B", "C", "D")]
    parts.append(external_table([("Reading", reading_ext)], "Table 3.1  Reading: external data (arms A-D)",
                                "QR", ict_role="qict"))
    groups = [(fmt.capitalize(), [find(typing_v2, f"{fmt}{suffix}") for suffix in ("", "_audio", "_visual", "_av")])
              for fmt in ("continuous", "keypress", "word")]
    parts.append("\n" + external_table(groups, "Table 3.2  Typing v2: external data (task-only rows are 3-seed means)",
                                       "Typing->resp."))
    groups = [(fmt.capitalize(), [find(typing_v1, f"{fmt}{suffix}") for suffix in ("", "_audio", "_visual", "_av")])
              for fmt in ("continuous", "keypress")]
    parts.append("\n" + external_table(groups, "Table 3.3  Typing v1: external data (ICT target = typed query)",
                                       "Typing->resp."))
    imag_ext = [find(imagining_arms, k) for k in ("A_position", "B", "C", "D")]
    parts.append("\n" + external_table([("Imagining", imag_ext)], "Table 3.4  Imagining: external data (arms A-D)",
                                       "Imagined->resp."))
    parts.append("  See findings/summary_figures/fig1_external_data_effect.png.\n")

    parts.append("""

4. RESULTS BY MODALITY
======================

How to read the tables
- "Full pooled index" tables: all test queries against all test documents of the arm.
  #Docs differs between arms, so compare each row with its own noise baseline.
- "MRR by masking level" tables give #Docs in brackets at each masking level.
- "Per-source" tables: each source against its own test documents, at 90% masking.

4.1 Reading
-----------
- Adding question ICT to query -> response (arm A) gives the best task result:
  query -> response {r_a_main} vs {r_a_noise} noise on its own index.
- Arm A also scores {r_a_pool} on the full pooled index vs {r_a_pool_noise} noise.
- With audio data (arm B), pooled MRR at 0% masking rises to {r_b_pool0} and Hit@10 to
  {r_b_h10}.

""".format(r_a_main=fv(find(reading, "A"), "main"), r_a_noise=fmt_val(find(reading, "A").get("main", "noise", "rr")),
           r_a_pool=fv(find(reading, "A"), "pooled"),
           r_a_pool_noise=fmt_val(find(reading, "A").get("pooled", "noise", "rr")),
           r_b_pool0=fmt_val(find(reading, "B").get("pooled", 0, "rr")),
           r_b_h10=fmt_val(find(reading, "B").get("pooled", 0, "hit_at_10"))))
    parts.append(pooled_at_90_table(reading, "Table 4.1  Reading: full pooled index, 90% masking"))
    parts.append("\n" + pooled_by_level_table(reading, "Table 4.2  Reading: MRR by masking level [#Docs]"))
    parts.append("\n" + pooled_by_level_table(reading, "Table 4.3  Reading: Hit@10 by masking level",
                                              metric="hit_at_10", with_docs=False))
    parts.append("\n" + per_source_table(reading, [("Query->response", "main"), ("Question ICT", "qict"),
                                                   ("Response ICT", "rict"), ("Alice", "alice"),
                                                   ("Nieuwland", "nieuwland")],
                                         "Table 4.4  Reading: per-source results"))
    parts.append("  Figures: summary_figures/fig5_reading_ablations.png, fig2_mrr_and_docs_vs_masking.png.\n")

    cont, key, word = (find(typing_v2, k) for k in ("continuous", "keypress", "word"))
    parts.append("""
4.2 Typing
----------
- Typing v2 (3 seeds) is stable across seeds. On the full pooled index at 90% masking:
  continuous {c} vs {cn} noise, keypress {k} vs {kn}, word {w} vs {wn}.
- Keypress typing gives the strongest typing ICT signal: {kict} vs {kictn} noise.
- With audio + visual data, pooled MRR reaches about 0.14 for all three formats.

""".format(c=fv(cont, "pooled"), cn=fmt_val(cont.get("pooled", "noise", "rr")), k=fv(key, "pooled"),
           kn=fmt_val(key.get("pooled", "noise", "rr")), w=fv(word, "pooled"),
           wn=fmt_val(word.get("pooled", "noise", "rr")), kict=fv(key, "ict"),
           kictn=fmt_val(key.get("ict", "noise", "rr"))))
    parts.append(pooled_at_90_table(typing_v2, "Table 4.5  Typing v2: full pooled index, 90% masking"))
    parts.append("\n" + pooled_by_level_table(typing_v2, "Table 4.6  Typing v2: MRR by masking level [#Docs]"))
    parts.append("\n" + per_source_table(typing_v2, [("Typing->response", "main"), ("Typing question ICT", "ict"),
                                                     ("Alice", "alice"), ("Nieuwland", "nieuwland")],
                                         "Table 4.7  Typing v2: per-source results"))
    parts.append("\n" + pooled_at_90_table(typing_v1, "Table 4.8  Typing v1: full pooled index, 90% masking"))
    parts.append("\n" + pooled_by_level_table(typing_v1, "Table 4.9  Typing v1: MRR by masking level [#Docs]"))
    parts.append("\n" + per_source_table(typing_v1, [("Typing->response", "main"), ("Typing ICT (typed query)", "ict"),
                                                     ("Alice", "alice"), ("Nieuwland", "nieuwland")],
                                         "Table 4.10 Typing v1: per-source results"))
    parts.append("  Figures: summary_figures/fig4_typing_formats.png, fig1, fig2.\n")

    parts.append("""
4.3 Imagining
-------------
- On imagining data alone, the eight methods score close to the noise baseline. The
  ~40-response index is small, so chance MRR is about 0.10.
- Adding imagining ICT with pace-aligned or random-span masking lifts the ICT above noise.
- Adding external data gives the best imagining results: imagined -> response
  {c} with visual data (noise {cn}) and {b} with audio (noise {bn}).

""".format(c=fv(find(imagining_arms, "C"), "main"), cn=fmt_val(find(imagining_arms, "C").get("main", "noise", "rr")),
           b=fv(find(imagining_arms, "B"), "main"), bn=fmt_val(find(imagining_arms, "B").get("main", "noise", "rr"))))
    parts.append(method_table(imagining_methods + imagining_arms,
                              "Table 4.11 Imagining: imagined EEG -> response, per-source index (unmasked)"))
    parts.append("\n" + pooled_at_90_table(imagining_arms, "Table 4.12 Imagining arms: full pooled index, 90% masking"))
    parts.append("\n" + pooled_by_level_table(imagining_arms, "Table 4.13 Imagining arms: MRR by masking level [#Docs]"))
    parts.append("\n" + per_source_table(imagining_arms, [("Imagined->response", "main"), ("Imagining ICT", "ict"),
                                                          ("Alice", "alice"), ("Nieuwland", "nieuwland")],
                                         "Table 4.14 Imagining arms: per-source results"))
    parts.append("  Method runs have one source and no masking, so their #Docs is the same at every level.\n")
    parts.append("  Figures: summary_figures/fig3_imagining_methods.png, fig1, fig2.\n")
    return "\n".join(parts)


def main():
    _style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    reading, typing_v1, typing_v2, imagining_methods, imagining_arms = build_runs()
    OUT_TXT.write_text(build_text(reading, typing_v1, typing_v2, imagining_methods, imagining_arms))
    print(OUT_TXT)
    for path in (fig_external_effect(reading, typing_v2, imagining_arms),
                 fig_masking_curves(reading, typing_v2, imagining_arms),
                 fig_imagining_methods(imagining_methods, imagining_arms),
                 fig_typing_formats(typing_v1, typing_v2),
                 fig_reading_ablation(reading)):
        print(path)


if __name__ == "__main__":
    main()
