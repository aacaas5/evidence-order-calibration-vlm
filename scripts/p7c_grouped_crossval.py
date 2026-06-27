import json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


SEED = 42
MU = 0.3
MARGIN = 0.05
L2 = 1.0
N_SPLITS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = Path("results/p6a/hidden_features.npz")
META = Path("results/p6a/hidden_features_meta.json")

OUT = Path("results/p7c")
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

X = np.load(FEATURES)["X"].astype(np.float32)

df = pd.DataFrame(
    json.loads(META.read_text(encoding="utf-8"))
)

y = df["correct"].astype(np.float32).to_numpy()
groups = df["question_id"].astype(str).to_numpy()
severity = df["severity"].astype(float).to_numpy()

print("Feature matrix:", X.shape)
print("Trajectories:", len(np.unique(groups)))


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class ReliabilityHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def emvr(probs, test_qids):

    violations = 0
    pairs = 0
    traj_bad = 0
    traj_total = 0

    for qid in test_qids:

        idx = np.where(groups == qid)[0]
        idx = idx[np.argsort(severity[idx])]

        if len(idx) != 5:
            continue

        traj_total += 1
        bad = 0

        for i in range(4):
            if probs[idx[i + 1]] > probs[idx[i]]:
                violations += 1
                bad += 1

            pairs += 1

        if bad:
            traj_bad += 1

    return (
        violations / pairs,
        traj_bad / traj_total
    )


def safe_auc(yy, pp):
    if len(np.unique(yy)) < 2:
        return float("nan")
    return roc_auc_score(yy, pp)


# ------------------------------------------------------------
# Train one fold
# ------------------------------------------------------------

def train_model(Xs, train_idx, use_order):

    torch.manual_seed(SEED)

    Xt = torch.tensor(
        Xs,
        dtype=torch.float32,
        device=DEVICE
    )

    yt = torch.tensor(
        y,
        dtype=torch.float32,
        device=DEVICE
    )

    model = ReliabilityHead(X.shape[1]).to(DEVICE)

    train_y = y[train_idx]

    n = len(train_y)
    n_pos = train_y.sum()
    n_neg = n - n_pos

    w_pos = n / (2 * n_pos)
    w_neg = n / (2 * n_neg)

    train_idx_t = torch.tensor(
        train_idx,
        dtype=torch.long,
        device=DEVICE
    )

    train_qids = np.unique(groups[train_idx])

    pairs = []

    for qid in train_qids:
        idx = np.where(groups == qid)[0]
        idx = idx[np.argsort(severity[idx])]

        if len(idx) == 5:
            for i in range(4):
                pairs.append((idx[i], idx[i + 1]))


    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=250,
        line_search_fn="strong_wolfe",
    )


    def closure():

        optimizer.zero_grad()

        logits = model(Xt)

        train_logits = logits[train_idx_t]
        train_targets = yt[train_idx_t]

        raw = nn.functional.binary_cross_entropy_with_logits(
            train_logits,
            train_targets,
            reduction="none",
        )

        weights = torch.where(
            train_targets > 0.5,
            torch.tensor(w_pos, device=DEVICE),
            torch.tensor(w_neg, device=DEVICE),
        )

        cls = (raw * weights).mean()

        reg = sum(
            (p ** 2).sum()
            for p in model.parameters()
        )

        loss = cls + L2 * reg

        if use_order:

            ord_losses = []

            for a, b in pairs:
                ord_losses.append(
                    torch.relu(
                        MARGIN - (logits[a] - logits[b])
                    )
                )

            loss = loss + MU * torch.stack(
                ord_losses
            ).mean()

        loss.backward()

        return loss


    optimizer.step(closure)

    model.eval()

    with torch.no_grad():
        logits = model(Xt).cpu().numpy()

    probs = 1 / (
        1 + np.exp(
            -np.clip(logits, -30, 30)
        )
    )

    return probs


# ------------------------------------------------------------
# Grouped cross-validation
# ------------------------------------------------------------

gkf = GroupKFold(n_splits=N_SPLITS)

rows = []

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups),
    start=1
):

    train_q = np.unique(groups[train_idx])
    test_q = np.unique(groups[test_idx])

    scaler = StandardScaler()

    Xs = np.empty_like(X)

    Xs[train_idx] = scaler.fit_transform(
        X[train_idx]
    )

    Xs[test_idx] = scaler.transform(
        X[test_idx]
    )


    print(
        f"\nFold {fold}/{N_SPLITS} | "
        f"train_q={len(train_q)} "
        f"test_q={len(test_q)}"
    )


    bce_probs = train_model(
        Xs,
        train_idx,
        use_order=False
    )

    order_probs = train_model(
        Xs,
        train_idx,
        use_order=True
    )


    for name, probs in [
        ("bce_only", bce_probs),
        ("bce_plus_order", order_probs),
    ]:

        yy = y[test_idx]
        pp = probs[test_idx]

        pred = (
            pp >= 0.5
        ).astype(int)

        fold_emvr, traj_rate = emvr(
            probs,
            test_q
        )

        row = {
            "fold": fold,
            "model": name,
            "test_questions": len(test_q),
            "test_conditions": len(test_idx),
            "accuracy": accuracy_score(
                yy, pred
            ),
            "auroc": safe_auc(
                yy, pp
            ),
            "auprc": average_precision_score(
                yy, pp
            ),
            "brier": brier_score_loss(
                yy, pp
            ),
            "emvr": fold_emvr,
            "trajectory_violation_rate": traj_rate,
        }

        rows.append(row)


results = pd.DataFrame(rows)

results.to_csv(
    OUT / "fold_results.csv",
    index=False
)


# ------------------------------------------------------------
# Aggregate results
# ------------------------------------------------------------

summary = (
    results.groupby("model")
    .agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"),
        auprc_mean=("auprc", "mean"),
        auprc_std=("auprc", "std"),
        brier_mean=("brier", "mean"),
        brier_std=("brier", "std"),
        emvr_mean=("emvr", "mean"),
        emvr_std=("emvr", "std"),
        traj_violation_mean=(
            "trajectory_violation_rate",
            "mean"
        ),
    )
    .reset_index()
)

summary.to_csv(
    OUT / "summary.csv",
    index=False
)


# ------------------------------------------------------------
# Fold-by-fold EMVR differences
# ------------------------------------------------------------

pivot = results.pivot(
    index="fold",
    columns="model",
    values="emvr"
)

pivot["delta_emvr_order_minus_bce"] = (
    pivot["bce_plus_order"]
    - pivot["bce_only"]
)

pivot.to_csv(
    OUT / "emvr_differences.csv"
)


print("\n" + "=" * 100)
print("P7C GROUPED CROSS-VALIDATION")
print("=" * 100)

print("\nFOLD RESULTS")

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSUMMARY")

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nEMVR DIFFERENCES")
print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))

print("\nP7C COMPLETE")
print("Saved:", OUT)
