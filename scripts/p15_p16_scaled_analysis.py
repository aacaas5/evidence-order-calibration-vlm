import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score


SEED = 42
N_BOOTSTRAP = 1000
PREDICTIONS = Path("results/scaled/p8/oof_predictions.csv")
QWEN_RESULTS = Path("results/scaled/p5/results.json")
MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_accepted.json")
IMAGE_DIR = Path("data/gqa/scaled_images")
STATS_OUT = Path("results/scaled/statistics")
FAILURE_OUT = Path("results/scaled/failure_analysis")
STATS_OUT.mkdir(parents=True, exist_ok=True)
FAILURE_OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS, dtype={"question_id": str, "image_id": str})
df["question_id"] = df["question_id"].astype(str)
df["correct"] = df["correct"].astype(bool)
score_columns = {
    "native_confidence": "native_confidence",
    "bce_only": "bce_oof",
    "bce_plus_order": "order_oof",
}
question_ids = np.asarray(sorted(df["question_id"].unique()))
blocks = {
    question_id: df.index[df["question_id"] == question_id].to_numpy()[
        np.argsort(df.loc[df["question_id"] == question_id, "severity"].to_numpy())
    ]
    for question_id in question_ids
}
if any(len(indices) != 5 for indices in blocks.values()):
    raise RuntimeError("Bootstrap requires complete five-condition trajectories")


def evaluate(block_sequence, score_column):
    indices = np.concatenate(block_sequence)
    scores = df.loc[indices, score_column].to_numpy(float)
    targets = df.loc[indices, "correct"].to_numpy(int)
    violations = 0
    for block in block_sequence:
        trajectory_scores = df.loc[block, score_column].to_numpy(float)
        violations += int(np.sum(trajectory_scores[1:] > trajectory_scores[:-1]))
    emvr = violations / (4 * len(block_sequence))
    order = np.argsort(-scores, kind="stable")
    sorted_targets = targets[order]
    coverage = np.arange(1, len(targets) + 1) / len(targets)
    risk = 1 - np.cumsum(sorted_targets) / np.arange(1, len(targets) + 1)
    aurc = float(np.trapezoid(risk, coverage))
    auroc = float(roc_auc_score(targets, scores)) if len(np.unique(targets)) == 2 else float("nan")
    return {"emvr": emvr, "aurc": aurc, "auroc": auroc}


rng = np.random.default_rng(SEED)
bootstrap = {
    model: {metric: [] for metric in ("emvr", "aurc", "auroc")}
    for model in score_columns
}
paired = {metric: [] for metric in ("emvr", "aurc", "auroc")}
all_blocks = [blocks[question_id] for question_id in question_ids]
point = {
    model: evaluate(all_blocks, column)
    for model, column in score_columns.items()
}

for iteration in range(N_BOOTSTRAP):
    sampled_ids = rng.choice(question_ids, size=len(question_ids), replace=True)
    sampled_blocks = [blocks[question_id] for question_id in sampled_ids]
    iteration_metrics = {}
    for model, column in score_columns.items():
        values = evaluate(sampled_blocks, column)
        iteration_metrics[model] = values
        for metric, value in values.items():
            bootstrap[model][metric].append(value)
    for metric in paired:
        paired[metric].append(
            iteration_metrics["bce_plus_order"][metric] - iteration_metrics["bce_only"][metric]
        )
    if (iteration + 1) % 200 == 0:
        print(f"Bootstrap {iteration + 1}/{N_BOOTSTRAP}")


def interval(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "lower_95": float(np.quantile(finite, 0.025)),
        "upper_95": float(np.quantile(finite, 0.975)),
        "bootstrap_mean": float(np.mean(finite)),
        "valid_resamples": int(len(finite)),
    }


summary = {
    "seed": SEED,
    "bootstrap_resamples": N_BOOTSTRAP,
    "resampling_unit": "question_trajectory",
    "models": {
        model: {
            metric: {"point_estimate": point[model][metric], **interval(bootstrap[model][metric])}
            for metric in ("emvr", "aurc", "auroc")
        }
        for model in score_columns
    },
    "paired_order_minus_bce": {
        metric: {
            "point_estimate": point["bce_plus_order"][metric] - point["bce_only"][metric],
            **interval(values),
        }
        for metric, values in paired.items()
    },
}
(STATS_OUT / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

# Failure analysis
qwen_rows = pd.DataFrame(json.loads(QWEN_RESULTS.read_text(encoding="utf-8")))
qwen_rows["question_id"] = qwen_rows["question_id"].astype(str)
manifest = {str(sample["question_id"]): sample for sample in json.loads(MANIFEST.read_text(encoding="utf-8"))}
trajectory_rows = []
for question_id in question_ids:
    trajectory = df[df["question_id"] == question_id].sort_values("severity")
    record = {
        "question_id": question_id,
        "image_id": str(trajectory.iloc[0]["image_id"]),
        "category": trajectory.iloc[0]["category"],
        "question": manifest[question_id]["question"],
        "ground_truth": manifest[question_id]["answer"],
        "clean_correct": bool(trajectory.iloc[0]["correct"]),
        "correctness_trajectory": "|".join("1" if value else "0" for value in trajectory["correct"]),
    }
    for model, column in score_columns.items():
        scores = trajectory[column].to_numpy(float)
        record[f"{model}_violations"] = int(np.sum(scores[1:] > scores[:-1]))
        record[f"{model}_max_increase"] = float(np.max(scores[1:] - scores[:-1]))
        record[f"{model}_scores"] = "|".join(f"{value:.6f}" for value in scores)
    answers = qwen_rows[qwen_rows["question_id"] == question_id].sort_values("severity")["answer"].astype(str)
    record["answer_trajectory"] = "|".join(answers)
    record["order_help"] = record["bce_only_violations"] - record["bce_plus_order_violations"]
    trajectory_rows.append(record)

trajectory_df = pd.DataFrame(trajectory_rows)
representatives = []


def add_cases(label, candidates, count=3):
    for _, row in candidates.head(count).iterrows():
        item = row.to_dict()
        item["analysis_type"] = label
        representatives.append(item)


add_cases(
    "strongest_native_failure",
    trajectory_df.sort_values(["native_confidence_violations", "native_confidence_max_increase"], ascending=False),
)
add_cases(
    "order_helps",
    trajectory_df[trajectory_df["order_help"] > 0].sort_values(["order_help", "bce_only_max_increase"], ascending=False),
)
add_cases(
    "order_hurts",
    trajectory_df[trajectory_df["order_help"] < 0].sort_values("order_help"),
)
add_cases(
    "native_confidence_strong",
    trajectory_df[(trajectory_df["native_confidence_violations"] == 0) & trajectory_df["clean_correct"]]
    .sort_values("native_confidence_max_increase"),
)
add_cases(
    "proposed_method_failure",
    trajectory_df.sort_values(["bce_plus_order_violations", "bce_plus_order_max_increase"], ascending=False),
)
representative_df = pd.DataFrame(representatives)
representative_df.to_csv(FAILURE_OUT / "representative_cases.csv", index=False)


def corrupt(image, box, severity):
    if severity == 0:
        return image.copy()
    x1, y1, x2, y2 = map(float, box)
    scale = math.sqrt(severity)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w, half_h = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    output = image.copy()
    ImageDraw.Draw(output).rectangle(
        [cx - half_w, cy - half_h, cx + half_w, cy + half_h], fill=(127, 127, 127)
    )
    return output


help_candidates = trajectory_df[trajectory_df["order_help"] > 0].sort_values("order_help", ascending=False)
failure_candidates = trajectory_df.sort_values(
    ["bce_plus_order_violations", "bce_plus_order_max_increase"], ascending=False
)
figure_cases = []
if not help_candidates.empty:
    figure_cases.append(("Order helps", help_candidates.iloc[0]))
figure_cases.append(("Proposed-method failure", failure_candidates.iloc[0]))

fig, axes = plt.subplots(len(figure_cases), 5, figsize=(18, 4.2 * len(figure_cases)), squeeze=False)
for row_index, (label, case) in enumerate(figure_cases):
    sample = manifest[str(case["question_id"])]
    original = Image.open(IMAGE_DIR / f'{sample["image_id"]}.jpg').convert("RGB")
    qwen_trajectory = qwen_rows[qwen_rows["question_id"] == str(case["question_id"])].sort_values("severity")
    prediction_trajectory = df[df["question_id"] == str(case["question_id"])].sort_values("severity")
    for column_index, severity_value in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
        axis = axes[row_index, column_index]
        axis.imshow(corrupt(original, sample["critical_objects"][0]["bbox_xyxy"], severity_value))
        qwen_row = qwen_trajectory.iloc[column_index]
        prediction_row = prediction_trajectory.iloc[column_index]
        axis.set_title(
            f"λ={severity_value:.2f}  ans={qwen_row['answer']}  ok={int(qwen_row['correct'])}\n"
            f"native={prediction_row['native_confidence']:.3f}  "
            f"BCE={prediction_row['bce_oof']:.3f}  order={prediction_row['order_oof']:.3f}",
            fontsize=9,
        )
        axis.axis("off")
    axes[row_index, 0].set_ylabel(
        f"{label}\nQ: {sample['question']}\nGT: {sample['answer']}", fontsize=9
    )
plt.tight_layout()
plt.savefig(FAILURE_OUT / "qualitative_cases.png", dpi=180)
plt.close()

failure_summary = {
    "trajectories": len(trajectory_df),
    "representative_rows": len(representative_df),
    "order_helps_count": int((trajectory_df["order_help"] > 0).sum()),
    "order_hurts_count": int((trajectory_df["order_help"] < 0).sum()),
    "order_equal_count": int((trajectory_df["order_help"] == 0).sum()),
    "outputs": [
        str(FAILURE_OUT / "representative_cases.csv"),
        str(FAILURE_OUT / "qualitative_cases.png"),
    ],
}
(FAILURE_OUT / "summary.json").write_text(json.dumps(failure_summary, indent=2), encoding="utf-8")

print("P15 BOOTSTRAP SUMMARY")
print(json.dumps(summary, indent=2))
print("P16 FAILURE SUMMARY")
print(json.dumps(failure_summary, indent=2))
print("P15-P16 COMPLETE")
