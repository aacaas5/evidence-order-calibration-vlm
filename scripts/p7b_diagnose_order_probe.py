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
    log_loss,
)
from sklearn.preprocessing import StandardScaler


# ============================================================
# Setup
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURES = Path("results/p6a/hidden_features.npz")
META = Path("results/p6a/hidden_features_meta.json")
SPLIT = Path("results/p6b/split.json")

OUT = Path("results/p7b")
OUT.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)


# ============================================================
# Load data
# ============================================================

X = np.load(FEATURES)["X"].astype(np.float32)
df = pd.DataFrame(json.loads(META.read_text(encoding="utf-8")))
split = json.loads(SPLIT.read_text(encoding="utf-8"))

y = df["correct"].astype(np.float32).to_numpy()
groups = df["question_id"].astype(str).to_numpy()
severity = df["severity"].astype(float).to_numpy()

train_q = np.array(split["train_question_ids"], dtype=str)
val_q   = np.array(split["val_question_ids"], dtype=str)
test_q  = np.array(split["test_question_ids"], dtype=str)

train_mask = np.isin(groups, train_q)
val_mask   = np.isin(groups, val_q)
test_mask  = np.isin(groups, test_q)

print("Feature matrix:", X.shape)
print("Train / Val / Test:",
      train_mask.sum(), val_mask.sum(), test_mask.sum())


# ============================================================
# Standardize using TRAIN statistics only
# ============================================================

scaler = StandardScaler()

X_scaled = np.empty_like(X)

X_scaled[train_mask] = scaler.fit_transform(X[train_mask])
X_scaled[val_mask]   = scaler.transform(X[val_mask])
X_scaled[test_mask]  = scaler.transform(X[test_mask])

Xt = torch.tensor(X_scaled, dtype=torch.float32, device=DEVICE)
yt = torch.tensor(y, dtype=torch.float32, device=DEVICE)


# ============================================================
# Balanced BCE weights
# ============================================================

train_y = y[train_mask]

n = len(train_y)
n_pos = train_y.sum()
n_neg = n - n_pos

w_pos = n / (2 * n_pos)
w_neg = n / (2 * n_neg)

print(f"Train positives: {int(n_pos)}")
print(f"Train negatives: {int(n_neg)}")
print(f"class weights: pos={w_pos:.4f}, neg={w_neg:.4f}")


# ============================================================
# Linear reliability head
# ============================================================

class ReliabilityHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# ============================================================
# Helpers
# ============================================================

def weighted_bce(logits, targets):
    raw = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )

    weights = torch.where(
        targets > 0.5,
        torch.tensor(w_pos, device=DEVICE),
        torch.tensor(w_neg, device=DEVICE),
    )

    return (raw * weights).mean()


def get_pairs(question_ids):
    pairs = []

    for qid in question_ids:
        idx = np.where(groups == qid)[0]
        idx = idx[np.argsort(severity[idx])]

        if len(idx) != 5:
            continue

        for i in range(4):
            pairs.append((int(idx[i]), int(idx[i + 1])))

    return pairs


train_pairs = get_pairs(train_q)


def ordinal_loss(logits, margin):
    losses = []

    for a, b in train_pairs:
        losses.append(
            torch.relu(
                margin - (logits[a] - logits[b])
            )
        )

    return torch.stack(losses).mean()


def emvr(probs, question_ids):
    violations = 0
    pairs = 0
    trajectories_bad = 0
    trajectories = 0

    for qid in question_ids:
        idx = np.where(groups == qid)[0]
        idx = idx[np.argsort(severity[idx])]

        if len(idx) != 5:
            continue

        trajectories += 1
        local_bad = 0

        for i in range(4):
            if probs[idx[i + 1]] > probs[idx[i]]:
                violations += 1
                local_bad += 1

            pairs += 1

        if local_bad:
            trajectories_bad += 1

    return (
        violations / pairs,
        trajectories_bad / trajectories
    )


def safe_auc(y_true, p):
    return (
        roc_auc_score(y_true, p)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )


# ============================================================
# Train one model with full-batch LBFGS
# ============================================================

def train_model(l2, mu=0.0, margin=0.10):

    torch.manual_seed(SEED)

    model = ReliabilityHead(X.shape[1]).to(DEVICE)

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=250,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    train_idx = torch.tensor(
        np.where(train_mask)[0],
        dtype=torch.long,
        device=DEVICE
    )

    def closure():
        optimizer.zero_grad()

        logits = model(Xt)

        cls = weighted_bce(
            logits[train_idx],
            yt[train_idx]
        )

        reg = sum(
            (p ** 2).sum()
            for p in model.parameters()
        )

        loss = cls + l2 * reg

        if mu > 0:
            loss = loss + mu * ordinal_loss(
                logits, margin
            )

        loss.backward()
        return loss

    optimizer.step(closure)

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, name, l2, mu=0.0, margin=0.0):

    model.eval()

    with torch.no_grad():
        logits = model(Xt).cpu().numpy()

    probs = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))

    row = {
        "model": name,
        "l2": l2,
        "mu": mu,
        "margin": margin,
    }

    for split_name, mask, qids in [
        ("train", train_mask, train_q),
        ("val", val_mask, val_q),
        ("test", test_mask, test_q),
    ]:
        yy = y[mask]
        pp = probs[mask]
        pred = (pp >= 0.5).astype(int)

        ev, tvr = emvr(probs, qids)

        row[f"{split_name}_accuracy"] = accuracy_score(yy, pred)
        row[f"{split_name}_auroc"] = safe_auc(yy, pp)
        row[f"{split_name}_auprc"] = average_precision_score(yy, pp)
        row[f"{split_name}_brier"] = brier_score_loss(yy, pp)
        row[f"{split_name}_logloss"] = log_loss(yy, pp, labels=[0, 1])
        row[f"{split_name}_emvr"] = ev
        row[f"{split_name}_traj_violation"] = tvr

    return row, probs


# ============================================================
# Phase A: choose L2 using validation BCE baseline only
# ============================================================

l2_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

baseline_rows = []

print("\nTesting BCE regularization...")

for l2 in l2_grid:

    model = train_model(
        l2=l2,
        mu=0.0
    )

    row, _ = evaluate(
        model,
        "bce_only",
        l2
    )

    baseline_rows.append(row)

    print(
        f"L2={l2:<6g} "
        f"val_Brier={row['val_brier']:.4f} "
        f"val_AUROC={row['val_auroc']:.4f} "
        f"val_EMVR={row['val_emvr']:.4f}"
    )


baseline_df = pd.DataFrame(baseline_rows)

# Choose regularization using validation Brier only
best_idx = baseline_df["val_brier"].idxmin()
best_l2 = float(baseline_df.loc[best_idx, "l2"])

print("\nSelected L2 from validation Brier:", best_l2)


# ============================================================
# Retrain fixed BCE reference
# ============================================================

bce_model = train_model(
    l2=best_l2,
    mu=0.0
)

bce_row, bce_probs = evaluate(
    bce_model,
    "bce_only",
    best_l2
)


# ============================================================
# Phase B: order configurations
# ============================================================

order_rows = []

for mu in [0.1, 0.3, 1.0]:
    for margin in [0.05, 0.10]:

        print(
            f"Training order model "
            f"mu={mu}, margin={margin}"
        )

        model = train_model(
            l2=best_l2,
            mu=mu,
            margin=margin
        )

        row, probs = evaluate(
            model,
            "bce_plus_order",
            best_l2,
            mu,
            margin
        )

        order_rows.append(row)

        np.save(
            OUT / f"probs_mu{mu}_m{margin}.npy",
            probs
        )


# ============================================================
# Results
# ============================================================

results = pd.DataFrame(
    [bce_row] + order_rows
)

results.to_csv(
    OUT / "matched_results.csv",
    index=False
)

baseline_df.to_csv(
    OUT / "l2_selection.csv",
    index=False
)

np.save(
    OUT / "bce_probs.npy",
    bce_probs
)

torch.save(
    bce_model.state_dict(),
    OUT / "bce_reference.pt"
)


cols = [
    "model",
    "l2",
    "mu",
    "margin",
    "train_auroc",
    "train_emvr",
    "val_auroc",
    "val_emvr",
    "test_auroc",
    "test_brier",
    "test_emvr",
    "test_traj_violation",
]

print("\n" + "=" * 100)
print("P7B DIAGNOSTIC RESULTS")
print("=" * 100)

print(
    results[cols].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:", OUT)
print("P7B COMPLETE")
