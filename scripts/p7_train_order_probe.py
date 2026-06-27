import json
import random
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
from sklearn.preprocessing import StandardScaler


# ============================================================
# Paths
# ============================================================

FEATURES = Path("results/p6a/hidden_features.npz")
META = Path("results/p6a/hidden_features_meta.json")
SPLIT = Path("results/p6b/split.json")

OUT = Path("results/p7")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Load data
# ============================================================

X = np.load(FEATURES)["X"].astype(np.float32)

meta = json.loads(
    META.read_text(encoding="utf-8")
)

df = pd.DataFrame(meta)

y = df["correct"].astype(np.float32).to_numpy()
groups = df["question_id"].astype(str).to_numpy()
severity = df["severity"].astype(float).to_numpy()

split = json.loads(
    SPLIT.read_text(encoding="utf-8")
)

train_q = np.array(
    split["train_question_ids"],
    dtype=str
)

val_q = np.array(
    split["val_question_ids"],
    dtype=str
)

test_q = np.array(
    split["test_question_ids"],
    dtype=str
)

train_mask = np.isin(groups, train_q)
val_mask = np.isin(groups, val_q)
test_mask = np.isin(groups, test_q)

print("Train:", train_mask.sum())
print("Val:", val_mask.sum())
print("Test:", test_mask.sum())


# ============================================================
# Standardize using TRAIN only
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X[train_mask]
).astype(np.float32)

X_val = scaler.transform(
    X[val_mask]
).astype(np.float32)

X_test = scaler.transform(
    X[test_mask]
).astype(np.float32)


X_all_scaled = scaler.transform(X).astype(np.float32)


# ============================================================
# Torch tensors
# ============================================================

Xtr = torch.tensor(X_train, device=device)
ytr = torch.tensor(
    y[train_mask],
    device=device
).unsqueeze(1)

Xva = torch.tensor(X_val, device=device)
yva = torch.tensor(
    y[val_mask],
    device=device
).unsqueeze(1)

Xte = torch.tensor(X_test, device=device)

Xall = torch.tensor(
    X_all_scaled,
    device=device
)


# ============================================================
# Linear reliability head
# ============================================================

class ReliabilityHead(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x)


# ============================================================
# Trajectory indices inside TRAIN
# ============================================================

train_global_indices = np.where(train_mask)[0]

trajectory_local = {}

for qid in train_q:

    global_idx = np.where(groups == qid)[0]

    global_idx = global_idx[
        np.argsort(severity[global_idx])
    ]

    local_idx = []

    for gi in global_idx:
        pos = np.where(
            train_global_indices == gi
        )[0]

        if len(pos):
            local_idx.append(int(pos[0]))

    if len(local_idx) == 5:
        trajectory_local[qid] = local_idx


print(
    "Train trajectories for order loss:",
    len(trajectory_local)
)


# ============================================================
# Order loss
# ============================================================

def order_loss(logits, margin=0.10):

    losses = []

    for idxs in trajectory_local.values():

        g = logits[idxs, 0]

        for i in range(4):

            losses.append(
                torch.relu(
                    margin - (g[i] - g[i + 1])
                )
            )

    if not losses:
        return torch.tensor(
            0.0,
            device=logits.device
        )

    return torch.stack(losses).mean()


# ============================================================
# Train function
# ============================================================

def train_model(
    use_order,
    mu=1.0,
    margin=0.10,
    epochs=1000
):

    torch.manual_seed(SEED)

    model = ReliabilityHead(
        X.shape[1]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3
    )

    bce = nn.BCEWithLogitsLoss()

    best_state = None
    best_val = float("inf")

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        logits = model(Xtr)

        loss_cls = bce(
            logits,
            ytr
        )

        loss_ord = order_loss(
            logits,
            margin
        )

        loss = loss_cls

        if use_order:
            loss = (
                loss_cls
                + mu * loss_ord
            )

        loss.backward()
        optimizer.step()


        # validation BCE only
        model.eval()

        with torch.no_grad():

            val_logits = model(Xva)

            val_loss = bce(
                val_logits,
                yva
            ).item()

        if val_loss < best_val:

            best_val = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }


    model.load_state_dict(best_state)

    return model


# ============================================================
# Metrics
# ============================================================

def evaluate(model, name):

    model.eval()

    with torch.no_grad():

        logits_all = (
            model(Xall)
            .squeeze(1)
            .cpu()
            .numpy()
        )

    probs_all = 1 / (
        1 + np.exp(-logits_all)
    )

    probs_test = probs_all[test_mask]
    y_test = y[test_mask]

    pred = (
        probs_test >= 0.5
    ).astype(int)

    metrics = {
        "model": name,
        "accuracy": float(
            accuracy_score(
                y_test,
                pred
            )
        ),
        "auroc": float(
            roc_auc_score(
                y_test,
                probs_test
            )
        ),
        "auprc": float(
            average_precision_score(
                y_test,
                probs_test
            )
        ),
        "brier": float(
            brier_score_loss(
                y_test,
                probs_test
            )
        ),
    }


    # --------------------------------------------------------
    # Evidence monotonicity violation rate on TEST trajectories
    # --------------------------------------------------------

    violations = 0
    pairs = 0
    violating_trajectories = 0
    complete = 0

    for qid in test_q:

        idx = np.where(
            groups == qid
        )[0]

        idx = idx[
            np.argsort(
                severity[idx]
            )
        ]

        if len(idx) != 5:
            continue

        complete += 1

        r = probs_all[idx]

        v = 0

        for i in range(4):

            if r[i + 1] > r[i]:
                violations += 1
                v += 1

            pairs += 1

        if v > 0:
            violating_trajectories += 1


    metrics["emvr"] = (
        violations / pairs
        if pairs else float("nan")
    )

    metrics[
        "trajectory_violation_rate"
    ] = (
        violating_trajectories / complete
        if complete else float("nan")
    )


    return (
        metrics,
        probs_all
    )


# ============================================================
# Train matched models
# ============================================================

print("\nTraining BCE-only...")

bce_model = train_model(
    use_order=False
)

print("Training BCE + order...")

order_model = train_model(
    use_order=True,
    mu=1.0,
    margin=0.10
)


# ============================================================
# Evaluate
# ============================================================

bce_metrics, bce_probs = evaluate(
    bce_model,
    "bce_only"
)

order_metrics, order_probs = evaluate(
    order_model,
    "bce_plus_order"
)


results = pd.DataFrame(
    [
        bce_metrics,
        order_metrics
    ]
)

print("\n" + "=" * 72)
print("P7 MATCHED RELIABILITY EXPERIMENT")
print("=" * 72)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# Save results
# ============================================================

results.to_csv(
    OUT / "metrics.csv",
    index=False
)

pred_df = df.copy()

pred_df["bce_reliability"] = (
    bce_probs
)

pred_df["order_reliability"] = (
    order_probs
)

pred_df.to_csv(
    OUT / "predictions.csv",
    index=False
)

torch.save(
    bce_model.state_dict(),
    OUT / "bce_only.pt"
)

torch.save(
    order_model.state_dict(),
    OUT / "bce_plus_order.pt"
)


print("\nSaved:", OUT)
print("P7 COMPLETE")
