import json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


SEED = 42
MU = 0.3
MARGIN = 0.05
L2 = 1.0
N_SPLITS = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

FEATURES = Path("results/p6a/hidden_features.npz")
META = Path("results/p6a/hidden_features_meta.json")

OUT = Path("results/p8")
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

native_conf = X[:, -2]

print("Conditions:", len(y))
print("Trajectories:", len(np.unique(groups)))


# ------------------------------------------------------------
# Reliability head
# ------------------------------------------------------------

class Head(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)


# ------------------------------------------------------------
# Train on one CV fold
# ------------------------------------------------------------

def train_fold(Xs, train_idx, use_order):

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

    model = Head(X.shape[1]).to(DEVICE)

    train_y = y[train_idx]

    n = len(train_y)
    pos = train_y.sum()
    neg = n - pos

    w_pos = n / (2 * pos)
    w_neg = n / (2 * neg)

    train_idx_t = torch.tensor(
        train_idx,
        dtype=torch.long,
        device=DEVICE
    )

    pairs = []

    for qid in np.unique(groups[train_idx]):

        idx = np.where(groups == qid)[0]
        idx = idx[np.argsort(severity[idx])]

        if len(idx) == 5:

            for i in range(4):
                pairs.append(
                    (idx[i], idx[i + 1])
                )


    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=250,
        line_search_fn="strong_wolfe"
    )


    def closure():

        optimizer.zero_grad()

        logits = model(Xt)

        lt = logits[train_idx_t]
        target = yt[train_idx_t]

        raw = nn.functional.binary_cross_entropy_with_logits(
            lt,
            target,
            reduction="none"
        )

        weights = torch.where(
            target > 0.5,
            torch.tensor(w_pos, device=DEVICE),
            torch.tensor(w_neg, device=DEVICE)
        )

        cls = (raw * weights).mean()

        reg = sum(
            (p ** 2).sum()
            for p in model.parameters()
        )

        loss = cls + L2 * reg

        if use_order:

            order_terms = []

            for a, b in pairs:

                order_terms.append(
                    torch.relu(
                        MARGIN -
                        (logits[a] - logits[b])
                    )
                )

            loss = (
                loss
                + MU * torch.stack(
                    order_terms
                ).mean()
            )

        loss.backward()

        return loss


    optimizer.step(closure)

    model.eval()

    with torch.no_grad():

        logits = model(Xt).cpu().numpy()

    return 1 / (
        1 + np.exp(
            -np.clip(logits, -30, 30)
        )
    )


# ------------------------------------------------------------
# Generate OUT-OF-FOLD predictions
# ------------------------------------------------------------

oof_bce = np.zeros(len(y), dtype=np.float32)
oof_order = np.zeros(len(y), dtype=np.float32)

gkf = GroupKFold(n_splits=N_SPLITS)

for fold, (train_idx, test_idx) in enumerate(
    gkf.split(X, y, groups),
    start=1
):

    scaler = StandardScaler()

    Xs = np.empty_like(X)

    Xs[train_idx] = scaler.fit_transform(
        X[train_idx]
    )

    Xs[test_idx] = scaler.transform(
        X[test_idx]
    )

    print(
        f"Fold {fold}/{N_SPLITS} | "
        f"test conditions={len(test_idx)}"
    )

    p_bce = train_fold(
        Xs,
        train_idx,
        False
    )

    p_order = train_fold(
        Xs,
        train_idx,
        True
    )

    oof_bce[test_idx] = p_bce[test_idx]
    oof_order[test_idx] = p_order[test_idx]


# ------------------------------------------------------------
# Risk-Coverage
# ------------------------------------------------------------

def risk_coverage(scores):

    order = np.argsort(-scores)

    correct_sorted = y[order]

    coverage = []
    risk = []

    for k in range(1, len(y) + 1):

        answered = correct_sorted[:k]

        coverage.append(
            k / len(y)
        )

        risk.append(
            1.0 - answered.mean()
        )

    coverage = np.array(coverage)
    risk = np.array(risk)

    aurc = np.trapezoid(
        risk,
        coverage
    )

    return coverage, risk, aurc


models = {
    "native_confidence": native_conf,
    "bce_only": oof_bce,
    "bce_plus_order": oof_order,
}


summary_rows = []
curves = {}


for name, score in models.items():

    coverage, risk, aurc = risk_coverage(
        score
    )

    curves[name] = (
        coverage,
        risk
    )

    row = {
        "model": name,
        "aurc": float(aurc),
    }

    # Risk at useful approximate coverages
    for target in [
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5
    ]:

        idx = np.argmin(
            np.abs(
                coverage - target
            )
        )

        row[
            f"risk_at_{int(target*100)}pct"
        ] = float(risk[idx])

    summary_rows.append(row)


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT / "risk_coverage_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Save OOF predictions
# ------------------------------------------------------------

pred = df.copy()

pred["native_confidence"] = native_conf
pred["bce_oof"] = oof_bce
pred["order_oof"] = oof_order

pred.to_csv(
    OUT / "oof_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

for name, (coverage, risk) in curves.items():

    plt.plot(
        coverage,
        risk,
        label=name
    )

plt.xlabel("Coverage")
plt.ylabel("Risk (error rate)")
plt.title("Selective Prediction: Risk-Coverage")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "risk_coverage.png",
    dpi=220
)

plt.close()


# ------------------------------------------------------------
# Print
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("P8 SELECTIVE PREDICTION")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

best = summary.loc[
    summary["aurc"].idxmin()
]

print(
    "\nBest AURC:",
    best["model"],
    f"{best['aurc']:.4f}"
)

print("\nLower AURC = better selective reliability.")
print("Saved:", OUT)
print("P8 COMPLETE")
