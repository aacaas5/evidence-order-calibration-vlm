import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

# ============================================================
# Fixed experiment settings
# ============================================================

SEED = 42

MU = 0.3
MARGIN = 0.05
L2 = 1.0

EPOCHS = 500
LR = 0.01

np.random.seed(SEED)
torch.manual_seed(SEED)

FEATURE_PATH = Path(
    "results/scaled/p6/hidden_features.npz"
)

META_PATH = Path(
    "results/scaled/p6/hidden_features_meta.json"
)

OUT = Path(
    "results/scaled/p20"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Load features
# ============================================================

z = np.load(
    FEATURE_PATH
)

if "X" in z:
    X = z["X"]
else:
    X = z[z.files[0]]

meta = pd.DataFrame(
    json.loads(
        META_PATH.read_text(
            encoding="utf-8"
        )
    )
)

meta["question_id"] = (
    meta["question_id"]
    .astype(str)
)

meta["severity"] = (
    meta["severity"]
    .astype(float)
)

meta["correct"] = (
    meta["correct"]
    .astype(int)
)


# ============================================================
# Align ordering
# ============================================================

order = np.lexsort(
    (
        meta["severity"].values,
        meta["question_id"].values
    )
)

X = X[order]

meta = (
    meta.iloc[order]
    .reset_index(drop=True)
)

print("Feature matrix:", X.shape)
print("Conditions:", len(meta))
print(
    "Trajectories:",
    meta["question_id"].nunique()
)


# ============================================================
# Feature definitions
# ============================================================

# Feature vector layout:
#
# 0:2048  = hidden representation
# 2048    = c_seq
# 2049    = entropy

FEATURE_SETS = {

    "confidence_only":
        [2048],

    "entropy_only":
        [2049],

    "confidence_entropy":
        [2048, 2049],

    "hidden_only":
        list(range(2048)),

    "hidden_plus_signals":
        list(range(2050)),
}


# ============================================================
# Linear probe
# ============================================================

class Probe(nn.Module):

    def __init__(self, d):
        super().__init__()

        self.linear = nn.Linear(
            d,
            1
        )

    def forward(self, x):

        return self.linear(
            x
        ).squeeze(-1)


# ============================================================
# Train
# ============================================================

def train_probe(
    Xtrain,
    ytrain,
    qids,
    severities,
    use_order=False
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    xt = torch.tensor(
        Xtrain,
        dtype=torch.float32,
        device=device
    )

    yt = torch.tensor(
        ytrain,
        dtype=torch.float32,
        device=device
    )


    model = Probe(
        Xtrain.shape[1]
    ).to(device)


    # Balanced BCE
    pos = max(
        int(ytrain.sum()),
        1
    )

    neg = max(
        len(ytrain) - pos,
        1
    )

    wp = (
        len(ytrain)
        / (2 * pos)
    )

    wn = (
        len(ytrain)
        / (2 * neg)
    )

    weights = np.where(
        ytrain == 1,
        wp,
        wn
    )

    wt = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )


    groups = {}

    for i, q in enumerate(qids):

        groups.setdefault(
            str(q),
            []
        ).append(i)


    best_loss = float("inf")
    best_state = None


    for _ in range(EPOCHS):

        optimizer.zero_grad()

        logits = model(
            xt
        )

        bce = (
            nn.functional
            .binary_cross_entropy_with_logits(
                logits,
                yt,
                reduction="none"
            )
        )

        bce = (
            bce * wt
        ).mean()


        l2 = sum(
            (p ** 2).sum()
            for p in model.parameters()
        )

        loss = (
            bce
            + L2
            * l2
            / Xtrain.shape[1]
        )


        # ----------------------------------------
        # Ordinal evidence loss
        # ----------------------------------------

        if use_order:

            order_terms = []

            for q, idxs in groups.items():

                idxs = sorted(
                    idxs,
                    key=lambda i:
                        severities[i]
                )

                if len(idxs) != 5:
                    continue

                for a, b in zip(
                    idxs[:-1],
                    idxs[1:]
                ):

                    order_terms.append(
                        torch.relu(
                            MARGIN
                            - (
                                logits[a]
                                - logits[b]
                            )
                        )
                    )


            if order_terms:

                order_loss = (
                    torch.stack(
                        order_terms
                    ).mean()
                )

                loss = (
                    loss
                    + MU
                    * order_loss
                )


        loss.backward()

        optimizer.step()


        value = float(
            loss.detach().cpu()
        )

        if value < best_loss:

            best_loss = value

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()

                for k, v
                in model.state_dict().items()
            }


    model.load_state_dict(
        best_state
    )

    return model


# ============================================================
# Prediction
# ============================================================

def predict(
    model,
    Xtest
):

    device = next(
        model.parameters()
    ).device

    xt = torch.tensor(
        Xtest,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        probability = torch.sigmoid(
            model(xt)
        )

    return (
        probability
        .cpu()
        .numpy()
    )


# ============================================================
# Evidence-order metrics
# ============================================================

def order_metrics(
    scores,
    df
):

    d = df.copy()

    d["score"] = scores

    violations = 0
    pairs = 0

    bad_trajectories = 0
    trajectories = 0


    for qid, g in d.groupby(
        "question_id"
    ):

        g = g.sort_values(
            "severity"
        )

        if len(g) != 5:
            continue

        trajectories += 1

        vals = (
            g["score"]
            .values
        )

        local_bad = False

        for a, b in zip(
            vals[:-1],
            vals[1:]
        ):

            if b > a:

                violations += 1
                local_bad = True

            pairs += 1


        if local_bad:
            bad_trajectories += 1


    return (
        violations / pairs,
        bad_trajectories
        / trajectories
    )


# ============================================================
# Standard metrics
# ============================================================

def evaluate(
    y,
    prob,
    df
):

    pred = (
        prob >= 0.5
    ).astype(int)

    emvr, tvr = order_metrics(
        prob,
        df
    )

    return {

        "accuracy":
            accuracy_score(
                y,
                pred
            ),

        "auroc":
            roc_auc_score(
                y,
                prob
            ),

        "auprc":
            average_precision_score(
                y,
                prob
            ),

        "brier":
            brier_score_loss(
                y,
                prob
            ),

        "emvr":
            emvr,

        "trajectory_violation_rate":
            tvr,
    }


# ============================================================
# Grouped 5-fold evaluation
# ============================================================

question_ids = np.array(
    sorted(
        meta[
            "question_id"
        ].unique()
    )
)

gkf = GroupKFold(
    n_splits=5
)

fold_results = []


for fold, (
    train_q_idx,
    test_q_idx
) in enumerate(
    gkf.split(
        question_ids,
        groups=question_ids
    ),
    1
):

    train_q = set(
        question_ids[
            train_q_idx
        ]
    )

    test_q = set(
        question_ids[
            test_q_idx
        ]
    )


    train_mask = (
        meta["question_id"]
        .isin(train_q)
        .values
    )

    test_mask = (
        meta["question_id"]
        .isin(test_q)
        .values
    )


    y_train = (
        meta.loc[
            train_mask,
            "correct"
        ]
        .values
    )

    y_test = (
        meta.loc[
            test_mask,
            "correct"
        ]
        .values
    )


    train_meta = (
        meta.loc[
            train_mask
        ]
        .reset_index(drop=True)
    )

    test_meta = (
        meta.loc[
            test_mask
        ]
        .reset_index(drop=True)
    )


    print(
        f"\nFold {fold}/5 "
        f"| train q={len(train_q)} "
        f"| test q={len(test_q)}"
    )


    # ----------------------------------------
    # Feature ablations
    # ----------------------------------------

    for feature_name, columns in (
        FEATURE_SETS.items()
    ):

        Xtrain = (
            X[
                train_mask
            ][:, columns]
        )

        Xtest = (
            X[
                test_mask
            ][:, columns]
        )


        scaler = StandardScaler()

        Xtrain = (
            scaler.fit_transform(
                Xtrain
            )
        )

        Xtest = (
            scaler.transform(
                Xtest
            )
        )


        # BCE baseline
        model = train_probe(
            Xtrain,
            y_train,
            train_meta[
                "question_id"
            ].values,
            train_meta[
                "severity"
            ].values,
            use_order=False
        )


        prob = predict(
            model,
            Xtest
        )


        metrics = evaluate(
            y_test,
            prob,
            test_meta
        )


        fold_results.append({
            "fold":
                fold,

            "model":
                feature_name,

            "objective":
                "bce",

            "feature_dim":
                len(columns),

            **metrics,
        })


        # ------------------------------------
        # Order-loss version only for
        # full reliability representation
        # ------------------------------------

        if (
            feature_name
            == "hidden_plus_signals"
        ):

            order_model = train_probe(
                Xtrain,
                y_train,
                train_meta[
                    "question_id"
                ].values,
                train_meta[
                    "severity"
                ].values,
                use_order=True
            )


            order_prob = predict(
                order_model,
                Xtest
            )


            metrics = evaluate(
                y_test,
                order_prob,
                test_meta
            )


            fold_results.append({
                "fold":
                    fold,

                "model":
                    "hidden_plus_signals",

                "objective":
                    "bce_plus_order",

                "feature_dim":
                    len(columns),

                **metrics,
            })


# ============================================================
# Aggregate
# ============================================================

fold_df = pd.DataFrame(
    fold_results
)


summary = (
    fold_df
    .groupby(
        [
            "model",
            "objective",
            "feature_dim"
        ]
    )
    .agg(

        accuracy_mean=(
            "accuracy",
            "mean"
        ),

        accuracy_std=(
            "accuracy",
            "std"
        ),

        auroc_mean=(
            "auroc",
            "mean"
        ),

        auroc_std=(
            "auroc",
            "std"
        ),

        auprc_mean=(
            "auprc",
            "mean"
        ),

        brier_mean=(
            "brier",
            "mean"
        ),

        emvr_mean=(
            "emvr",
            "mean"
        ),

        emvr_std=(
            "emvr",
            "std"
        ),

        trajectory_violation_rate_mean=(
            "trajectory_violation_rate",
            "mean"
        )

    )
    .reset_index()
)


# ============================================================
# Save
# ============================================================

fold_df.to_csv(
    OUT
    / "p20a_ablation_fold_results.csv",
    index=False
)

summary.to_csv(
    OUT
    / "p20a_ablation_summary.csv",
    index=False
)


# ============================================================
# Print
# ============================================================

print("\n" + "=" * 110)

print(
    "P20A FINAL FEATURE / METHOD ABLATION"
)

print("=" * 110)


print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


print(
    "\nSaved:",
    OUT
    / "p20a_ablation_summary.csv"
)

print(
    "Saved:",
    OUT
    / "p20a_ablation_fold_results.csv"
)

print("\nP20A COMPLETE")
