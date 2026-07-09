import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn


SEED = 42
MU = 0.3
MARGIN = 0.05
L2 = 1.0
N_SPLITS = 5
FEATURES = Path("results/scaled/p6/hidden_features.npz")
META = Path("results/scaled/p6/hidden_features_meta.json")
P6_OUT = Path("results/scaled/p6")
P7_OUT = Path("results/scaled/p7")
P8_OUT = Path("results/scaled/p8")
for directory in (P6_OUT, P7_OUT, P8_OUT):
    directory.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = np.load(FEATURES)["X"].astype(np.float32)
df = pd.DataFrame(json.loads(META.read_text(encoding="utf-8")))
y = df["correct"].astype(np.float32).to_numpy()
groups = df["question_id"].astype(str).to_numpy()
severity = df["severity"].astype(float).to_numpy()
if X.shape != (len(df), 2050):
    raise RuntimeError(f"Unexpected input shape: X={X.shape}, rows={len(df)}")
if any(np.sum(groups == qid) != 5 for qid in np.unique(groups)):
    raise RuntimeError("Every question_id must have exactly five severity rows")


class ReliabilityHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.linear = nn.Linear(dimension, 1)

    def forward(self, features):
        return self.linear(features).squeeze(1)


def safe_auroc(targets, probabilities):
    return float(roc_auc_score(targets, probabilities)) if len(np.unique(targets)) > 1 else float("nan")


def trajectory_metrics(probabilities, question_ids):
    violations = pairs = bad_trajectories = trajectories = 0
    for question_id in question_ids:
        indices = np.where(groups == question_id)[0]
        indices = indices[np.argsort(severity[indices])]
        if len(indices) != 5 or np.any(~np.isfinite(probabilities[indices])):
            continue
        trajectory_bad = False
        for left, right in zip(indices[:-1], indices[1:]):
            violation = probabilities[right] > probabilities[left]
            violations += int(violation)
            pairs += 1
            trajectory_bad |= violation
        bad_trajectories += int(trajectory_bad)
        trajectories += 1
    return violations / pairs, bad_trajectories / trajectories


def train_head(features, train_idx, use_order):
    torch.manual_seed(SEED)
    tensor_x = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    tensor_y = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    train_tensor = torch.tensor(train_idx, dtype=torch.long, device=DEVICE)
    model = ReliabilityHead(features.shape[1]).to(DEVICE)
    train_y = y[train_idx]
    positive = float(train_y.sum())
    negative = float(len(train_y) - positive)
    if positive == 0 or negative == 0:
        raise RuntimeError("Balanced BCE requires both classes in every training fold")
    positive_weight = len(train_y) / (2 * positive)
    negative_weight = len(train_y) / (2 * negative)

    pairs = []
    if use_order:
        for question_id in np.unique(groups[train_idx]):
            indices = np.where(groups == question_id)[0]
            indices = indices[np.argsort(severity[indices])]
            if len(indices) == 5:
                pairs.extend(zip(indices[:-1], indices[1:]))

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=250, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        logits = model(tensor_x)
        targets = tensor_y[train_tensor]
        raw_bce = nn.functional.binary_cross_entropy_with_logits(
            logits[train_tensor], targets, reduction="none"
        )
        weights = torch.where(
            targets > 0.5,
            torch.tensor(positive_weight, device=DEVICE),
            torch.tensor(negative_weight, device=DEVICE),
        )
        loss = (raw_bce * weights).mean()
        loss = loss + L2 * sum((parameter ** 2).sum() for parameter in model.parameters())
        if use_order:
            order_loss = torch.stack([
                torch.relu(MARGIN - (logits[left] - logits[right]))
                for left, right in pairs
            ]).mean()
            loss = loss + MU * order_loss
        loss.backward()
        return loss

    optimizer.step(closure)
    model.eval()
    with torch.no_grad():
        logits = model(tensor_x).cpu().numpy()
    return 1 / (1 + np.exp(-np.clip(logits, -30, 30)))


def fold_metrics(fold, model_name, test_idx, probabilities):
    targets = y[test_idx]
    predicted = (probabilities[test_idx] >= 0.5).astype(int)
    emvr, trajectory_rate = trajectory_metrics(probabilities, np.unique(groups[test_idx]))
    return {
        "fold": fold,
        "model": model_name,
        "test_questions": len(np.unique(groups[test_idx])),
        "test_conditions": len(test_idx),
        "accuracy": accuracy_score(targets, predicted),
        "auroc": safe_auroc(targets, probabilities[test_idx]),
        "auprc": average_precision_score(targets, probabilities[test_idx]),
        "brier": brier_score_loss(targets, probabilities[test_idx]),
        "emvr": emvr,
        "trajectory_violation_rate": trajectory_rate,
    }


feature_sets = {
    "confidence_only": X[:, -2:-1],
    "confidence_entropy": X[:, -2:],
    "hidden_plus_signals": X,
}
oof = {name: np.full(len(y), np.nan, dtype=np.float32) for name in feature_sets}
oof["bce_plus_order"] = np.full(len(y), np.nan, dtype=np.float32)
p12_rows = []
p13_rows = []
splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups))

print("P12-P14 SCALED MODELS")
print("Conditions:", len(y), "Trajectories:", len(np.unique(groups)))
print("Frozen hyperparameters:", {"mu": MU, "margin": MARGIN, "l2": L2, "seed": SEED})

for fold, (train_idx, test_idx) in enumerate(splits, 1):
    print(f"Fold {fold}/{N_SPLITS}: train_q={len(np.unique(groups[train_idx]))} test_q={len(np.unique(groups[test_idx]))}")
    scaled_features = {}
    for model_name, features in feature_sets.items():
        scaler = StandardScaler()
        transformed = np.empty_like(features, dtype=np.float32)
        transformed[train_idx] = scaler.fit_transform(features[train_idx])
        transformed[test_idx] = scaler.transform(features[test_idx])
        scaled_features[model_name] = transformed
        probabilities = train_head(transformed, train_idx, use_order=False)
        oof[model_name][test_idx] = probabilities[test_idx]
        metrics = fold_metrics(fold, model_name, test_idx, probabilities)
        p12_rows.append(metrics)
        if model_name == "hidden_plus_signals":
            p13_rows.append({**metrics, "model": "bce_only"})

    order_probabilities = train_head(
        scaled_features["hidden_plus_signals"], train_idx, use_order=True
    )
    oof["bce_plus_order"][test_idx] = order_probabilities[test_idx]
    p13_rows.append(fold_metrics(fold, "bce_plus_order", test_idx, order_probabilities))

if any(np.any(~np.isfinite(values)) for values in oof.values()):
    raise RuntimeError("OOF prediction arrays contain missing values")

p12 = pd.DataFrame(p12_rows)
p12.to_csv(P6_OUT / "fold_results.csv", index=False)
p12_summary = p12.groupby("model").agg(
    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
    auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
    auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
    brier_mean=("brier", "mean"), brier_std=("brier", "std"),
    emvr_mean=("emvr", "mean"), emvr_std=("emvr", "std"),
    trajectory_violation_rate_mean=("trajectory_violation_rate", "mean"),
).reset_index()
p12_summary.to_csv(P6_OUT / "summary.csv", index=False)

p13 = pd.DataFrame(p13_rows)
p13.to_csv(P7_OUT / "fold_results.csv", index=False)
p13_summary = p13.groupby("model").agg(
    accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
    auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
    auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"),
    brier_mean=("brier", "mean"), brier_std=("brier", "std"),
    emvr_mean=("emvr", "mean"), emvr_std=("emvr", "std"),
    trajectory_violation_rate_mean=("trajectory_violation_rate", "mean"),
).reset_index()
p13_summary.to_csv(P7_OUT / "summary.csv", index=False)
pivot = p13.pivot(index="fold", columns="model", values="emvr")
pivot["delta_emvr_order_minus_bce"] = pivot["bce_plus_order"] - pivot["bce_only"]
pivot.to_csv(P7_OUT / "emvr_differences.csv")


def risk_coverage(scores):
    order = np.argsort(-scores, kind="stable")
    correct = y[order]
    coverage = np.arange(1, len(y) + 1) / len(y)
    risk = 1 - np.cumsum(correct) / np.arange(1, len(y) + 1)
    return coverage, risk, float(np.trapezoid(risk, coverage))


selective_scores = {
    "native_confidence": X[:, -2],
    "bce_only": oof["hidden_plus_signals"],
    "bce_plus_order": oof["bce_plus_order"],
}
curves = {}
risk_rows = []
for model_name, scores in selective_scores.items():
    coverage, risk, aurc = risk_coverage(scores)
    curves[model_name] = (coverage, risk)
    row = {"model": model_name, "aurc": aurc}
    for target in (0.9, 0.8, 0.7, 0.6, 0.5):
        index = int(np.argmin(np.abs(coverage - target)))
        row[f"risk_at_{int(target * 100)}pct"] = float(risk[index])
    risk_rows.append(row)
pd.DataFrame(risk_rows).to_csv(P8_OUT / "risk_coverage_summary.csv", index=False)

predictions = df.copy()
predictions["native_confidence"] = selective_scores["native_confidence"]
predictions["bce_oof"] = selective_scores["bce_only"]
predictions["order_oof"] = selective_scores["bce_plus_order"]
predictions.to_csv(P8_OUT / "oof_predictions.csv", index=False)

plt.figure(figsize=(7, 5))
for model_name, (coverage, risk) in curves.items():
    plt.plot(coverage, risk, label=model_name)
plt.xlabel("Coverage")
plt.ylabel("Risk (error rate)")
plt.title("Scaled Selective Prediction")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(P8_OUT / "risk_coverage.png", dpi=220)
plt.close()

print("\nP12 SUMMARY")
print(p12_summary.to_string(index=False))
print("\nP13 SUMMARY")
print(p13_summary.to_string(index=False))
print("\nP14 SUMMARY")
print(pd.DataFrame(risk_rows).to_string(index=False))
print("P12-P14 COMPLETE")
print("P12:", P6_OUT)
print("P13:", P7_OUT)
print("P14:", P8_OUT)
