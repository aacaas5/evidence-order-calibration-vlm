"""Regenerate paper figures exclusively from frozen cached outputs."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SEVERITIES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
RED, BLUE, GREEN, GOLD, GRAY = "#B2182B", "#2166AC", "#1B7837", "#B8860B", "#555555"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def finish(path, figure):
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def severity_curves():
    summary = json.loads((ROOT / "results/scaled/p5/summary.json").read_text(encoding="utf-8"))
    values = [summary["severity_summary"][str(float(s))] for s in SEVERITIES]
    specs = [
        ("accuracy", "Accuracy", RED, "o", (0, 0.72)),
        ("mean_c_seq", "Mean sequence log-confidence", BLUE, "s", None),
        ("mean_entropy", "Mean token entropy", GREEN, "^", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for label, (field, title, color, marker, ylim) in zip("abc", specs):
        axis = axes[ord(label) - ord("a")]
        axis.plot(SEVERITIES, [row[field] for row in values], color=color, marker=marker, lw=2)
        axis.set_title(f"({label}) {title}")
        axis.set_xlabel("Mask severity $\\lambda$")
        axis.set_xticks(SEVERITIES)
        if ylim:
            axis.set_ylim(*ylim)
        axis.grid(alpha=0.22)
    finish(OUT / "critical_mask_severity.png", fig)


def matched_control():
    metrics = json.loads((ROOT / "results/scaled/p18/control_metrics.json").read_text(encoding="utf-8"))
    specs = [("accuracy", "Accuracy"), ("mean_c_seq", "Mean $c_{seq}$"), ("mean_entropy", "Mean entropy")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for index, (field, title) in enumerate(specs):
        axis = axes[index]
        for intervention, label, color, marker in (
            ("critical", "Critical", RED, "o"),
            ("irrelevant", "Matched non-critical", BLUE, "s"),
        ):
            series = [metrics[intervention]["severity_summary"][str(float(s))][field] for s in SEVERITIES]
            axis.plot(SEVERITIES, series, label=label, color=color, marker=marker, lw=2)
        axis.set_title(f"({chr(97 + index)}) {title}")
        axis.set_xlabel("Mask severity $\\lambda$")
        axis.set_xticks(SEVERITIES)
        axis.grid(alpha=0.22)
    axes[0].legend(frameon=False, loc="lower left")
    finish(OUT / "matched_control.png", fig)


def method_comparison():
    masking = pd.read_csv(ROOT / "results/scaled/p20/paper_primary_masking.csv")
    transfer = pd.read_csv(ROOT / "results/scaled/p20/paper_primary_transfer.csv")
    mask_values = [
        float(masking.loc[masking.model == model, "emvr"].iloc[0])
        for model in ("native_confidence", "bce_only", "bce_plus_order")
    ]
    blur_values = [
        float(transfer.loc[transfer.model == model, "emvr_mean"].iloc[0])
        for model in ("native_confidence", "bce_only", "bce_plus_order")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    names = ["Native", "BCE", "BCE + order"]
    colors = [GRAY, BLUE, RED]
    for axis, title, values in zip(axes, ["(a) Critical masking", "(b) Mask-train $\\rightarrow$ blur-test"], [mask_values, blur_values]):
        bars = axis.bar(names, values, color=colors, width=0.68)
        axis.set_title(title)
        axis.set_ylim(0, 0.50)
        axis.set_ylabel("EMVR (lower is better)")
        axis.grid(axis="y", alpha=0.22)
        axis.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.009, f"{value:.3f}", ha="center", fontsize=9)
    finish(OUT / "emvr_comparison.png", fig)


def risk_coverage():
    frame = pd.read_csv(ROOT / "results/scaled/p8/oof_predictions.csv")
    specs = [
        ("native_confidence", "Native confidence", GRAY, "o"),
        ("bce_oof", "BCE", BLUE, "s"),
        ("order_oof", "BCE + order", RED, "^"),
    ]
    fig, axis = plt.subplots(figsize=(5.0, 3.4))
    for column, label, color, marker in specs:
        ordered = frame.sort_values(column, ascending=False, kind="stable")
        targets = ordered["correct"].astype(int).to_numpy()
        coverage = np.arange(1, len(targets) + 1) / len(targets)
        risk = 1 - np.cumsum(targets) / np.arange(1, len(targets) + 1)
        axis.plot(coverage, risk, label=label, color=color, lw=2, marker=marker, markevery=110, ms=4)
    axis.set(xlabel="Coverage", ylabel="Selective risk", xlim=(0, 1), ylim=(0, 0.65))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    finish(OUT / "risk_coverage.png", fig)


def pipeline():
    fig, axis = plt.subplots(figsize=(10.2, 2.25))
    axis.set_xlim(0, 10.2)
    axis.set_ylim(0, 2.25)
    axis.axis("off")
    boxes = [
        (0.15, "Question + image\ncritical region $M_q$", "#E8EEF5"),
        (2.25, "Five evidence states\n$\\lambda=0, .25, .5, .75, 1$", "#F7E9E8"),
        (4.45, "Frozen Qwen2.5-VL\nanswer + features", "#E8EEF5"),
        (6.55, "Reliability head\n$[h;c_{seq};H] \\rightarrow R_\\phi$", "#EAF3E8"),
        (8.55, "BCE + order loss\nEMVR evaluation", "#F7F0DC"),
    ]
    for x, text, face in boxes:
        patch = FancyBboxPatch((x, 0.65), 1.55, 0.95, boxstyle="round,pad=0.04,rounding_size=0.05",
                               facecolor=face, edgecolor="#444444", linewidth=1)
        axis.add_patch(patch)
        axis.text(x + 0.775, 1.125, text, ha="center", va="center", fontsize=9)
    for x in (1.72, 3.82, 6.02, 8.12):
        axis.add_patch(FancyArrowPatch((x, 1.125), (x + 0.48, 1.125), arrowstyle="-|>", mutation_scale=12,
                                       color="#444444", linewidth=1.2))
    axis.text(3.05, 0.25, "Trajectories supervise ordering during training; deployment uses one current state.",
              ha="left", va="center", fontsize=9, color="#333333")
    finish(OUT / "pipeline.png", fig)


def corrupt(image, box, severity):
    output = image.copy()
    if severity == 0:
        return output
    x1, y1, x2, y2 = map(float, box)
    scale = math.sqrt(severity)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    ImageDraw.Draw(output).rectangle([cx - half_w, cy - half_h, cx + half_w, cy + half_h], fill=(127, 127, 127))
    return output


def qualitative_trajectory():
    question_id = "16161462"
    manifest = {str(row["question_id"]): row for row in json.loads(
        (ROOT / "data/gqa/manifests/gqa_evidence_scaled_accepted.json").read_text(encoding="utf-8")
    )}
    rows = pd.DataFrame(json.loads((ROOT / "results/scaled/p5/results.json").read_text(encoding="utf-8")))
    trajectory = rows[rows.question_id.astype(str) == question_id].sort_values("severity")
    sample = manifest[question_id]
    image = Image.open(ROOT / "data/gqa/scaled_images" / f"{sample['image_id']}.jpg").convert("RGB")
    box = sample["critical_objects"][0]["bbox_xyxy"]
    fig, axes = plt.subplots(1, 5, figsize=(10.2, 2.35))
    for axis, (_, row) in zip(axes, trajectory.iterrows()):
        axis.imshow(corrupt(image, box, float(row.severity)))
        axis.set_title(
            f"$\\lambda$={row.severity:.2f}\n{row.answer} ({'correct' if row.correct else 'wrong'})\n$c_{{seq}}$={row.c_seq:.3f}",
            fontsize=8.5,
        )
        axis.axis("off")
    fig.suptitle(f"Q: {sample['question']}   Ground truth: {sample['answer']}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "qualitative_violation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


for function in (pipeline, severity_curves, matched_control, method_comparison, risk_coverage, qualitative_trajectory):
    function()
    print("generated", function.__name__)
