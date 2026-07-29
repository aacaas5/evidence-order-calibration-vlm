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

SEED = 42
MU = 0.3
MARGIN = 0.05
L2 = 1.0
EPOCHS = 500
LR = 0.01

np.random.seed(SEED)
torch.manual_seed(SEED)

MASK_X_PATH = Path(
    "results/scaled/p6/hidden_features.npz"
)

MASK_META_PATH = Path(
    "results/scaled/p6/hidden_features_meta.json"
)

BLUR_X_PATH = Path(
    "results/scaled/p19/blur_hidden_features.npz"
)

BLUR_META_PATH = Path(
    "results/scaled/p19/blur_hidden_meta.json"
)

OUT = Path(
    "results/scaled/p20"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Helpers
# ============================================================

def load_npz(path):
    z = np.load(path)

    if "X" in z:
        return z["X"]

    return z[z.files[0]]


def prepare_meta(path):

    df = pd.DataFrame(
        json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    )

    df["question_id"] = (
        df["question_id"]
        .astype(str)
    )

    df["severity"] = (
        df["severity"]
        .astype(float)
    )

    df["correct"] = (
        df["correct"]
        .astype(int)
    )

    return df


X_mask = load_npz(
    MASK_X_PATH
)

X_blur = load_npz(
    BLUR_X_PATH
)

meta_mask = prepare_meta(
    MASK_META_PATH
)

meta_blur = prepare_meta(
    BLUR_META_PATH
)


# ============================================================
# Sort consistently
# ============================================================

mask_order = np.lexsort(
    (
        meta_mask["severity"].values,
        meta_mask["question_id"].values
    )
)

blur_order = np.lexsort(
    (
        meta_blur["severity"].values,
        meta_blur["question_id"].values
    )
)

X_mask = X_mask[
    mask_order
]

meta_mask = (
    meta_mask.iloc[
        mask_order
    ]
    .reset_index(drop=True)
)

X_blur = X_blur[
    blur_order
]

meta_blur = (
    meta_blur.iloc[
        blur_order
    ]
    .reset_index(drop=True)
)


common_qids = sorted(
    set(
        meta_mask["question_id"]
    )
    &
    set(
        meta_blur["question_id"]
    )
)

print(
    "Mask conditions:",
    len(meta_mask)
)

print(
    "Blur conditions:",
    len(meta_blur)
)

print(
    "Common trajectories:",
    len(common_qids)
)

print(
    "Mask feature matrix:",
    X_mask.shape
)

print(
    "Blur feature matrix:",
    X_blur.shape
)


# ============================================================
# Feature sets
# ============================================================

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
# Probe
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


def train_probe(
    X,
    y,
    qids,
    severities,
    use_order=False
):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    Xt = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    yt = torch.tensor(
        y,
        dtype=torch.float32,
        device=device
    )

    model = Probe(
        X.shape[1]
    ).to(device)

    n_pos = max(
        int(y.sum()),
        1
    )

    n_neg = max(
        len(y) - n_pos,
        1
    )

    w_pos = (
        len(y)
        / (2 * n_pos)
    )

    w_neg = (
        len(y)
        / (2 * n_neg)
    )

    sample_weights = np.where(
        y == 1,
        w_pos,
        w_neg
    )

    wt = torch.tensor(
        sample_weights,
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
            Xt
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


        l2_penalty = sum(
            (p ** 2).sum()
            for p in model.parameters()
        )

        loss = (
            bce
            + L2
            * l2_penalty
            / X.shape[1]
        )


        if use_order:

            order_terms = []

            for q, idxs in groups.items():

                idxs = sorted(
                    idxs,
                    key=lambda j:
                        severities[j]
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
                    + MU * order_loss
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


def predict(
    model,
    X
):

    device = next(
        model.parameters()
    ).device

    Xt = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        p = torch.sigmoid(
            model(Xt)
        )

    return p.cpu().numpy()


# ============================================================
# Metrics
# ============================================================

def order_metrics(
    scores,
    df
):

    d = df.copy()
    d["score"] = scores

    violations = 0
    pairs = 0

    bad_traj = 0
    traj_n = 0


    for _, g in d.groupby(
        "question_id"
    ):

        g = g.sort_values(
            "severity"
        )

        if len(g) != 5:
            continue

        traj_n += 1

        vals = g[
            "score"
        ].values

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
            bad_traj += 1


    return (
        violations / pairs,
        bad_traj / traj_n
    )


def evaluate(
    y,
    probs,
    df
):

    pred = (
        probs >= 0.5
    ).astype(int)

    emvr, tvr = order_metrics(
        probs,
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
                probs
            ),

        "auprc":
            average_precision_score(
                y,
                probs
            ),

        "brier":
            brier_score_loss(
                y,
                probs
            ),

        "emvr":
            emvr,

        "trajectory_violation_rate":
            tvr,
    }


# ============================================================
# Grouped mask-train -> blur-test
# ============================================================

qids = np.array(
    common_qids
)

gkf = GroupKFold(
    n_splits=5
)

rows = []


for fold, (
    train_idx,
    test_idx
) in enumerate(
    gkf.split(
        qids,
        groups=qids
    ),
    1
):

    train_q = set(
        qids[
            train_idx
        ]
    )

    test_q = set(
        qids[
            test_idx
        ]
    )


    mask_train = (
        meta_mask[
            "question_id"
        ]
        .isin(train_q)
        .values
    )

    blur_test = (
        meta_blur[
            "question_id"
        ]
        .isin(test_q)
        .values
    )


    y_train = (
        meta_mask.loc[
            mask_train,
            "correct"
        ]
        .values
    )

    y_test = (
        meta_blur.loc[
            blur_test,
            "correct"
        ]
        .values
    )


    train_meta = (
        meta_mask.loc[
            mask_train
        ]
        .reset_index(drop=True)
    )

    test_meta = (
        meta_blur.loc[
            blur_test
        ]
        .reset_index(drop=True)
    )


    print(
        f"\nFold {fold}/5 "
        f"| mask train q={len(train_q)} "
        f"| blur test q={len(test_q)}"
    )


    for feature_name, columns in (
        FEATURE_SETS.items()
    ):

        Xm = (
            X_mask[
                mask_train
            ][:, columns]
        )

        Xb = (
            X_blur[
                blur_test
            ][:, columns]
        )


        scaler = StandardScaler()

        Xm = scaler.fit_transform(
            Xm
        )

        Xb = scaler.transform(
            Xb
        )


        # ----------------------------------------
        # BCE
        # ----------------------------------------

        model = train_probe(
            Xm,
            y_train,
            train_meta[
                "question_id"
            ].values,
            train_meta[
                "severity"
            ].values,
            use_order=False
        )

        probs = predict(
            model,
            Xb
        )

        metrics = evaluate(
            y_test,
            probs,
            test_meta
        )

        rows.append({
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


        # ----------------------------------------
        # Full features + order
        # ----------------------------------------

        if (
            feature_name
            == "hidden_plus_signals"
        ):

            order_model = train_probe(
                Xm,
                y_train,
                train_meta[
                    "question_id"
                ].values,
                train_meta[
                    "severity"
                ].values,
                use_order=True
            )

            order_probs = predict(
                order_model,
                Xb
            )

            metrics = evaluate(
                y_test,
                order_probs,
                test_meta
            )

            rows.append({
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
    rows
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
    / "p20b_transfer_ablation_fold_results.csv",
    index=False
)

summary.to_csv(
    OUT
    / "p20b_transfer_ablation_summary.csv",
    index=False
)


# ============================================================
# Print
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "P20B MASK-TRAIN -> BLUR-TEST FEATURE ABLATION"
)

print(
    "=" * 110
)


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
    / "p20b_transfer_ablation_summary.csv"
)

print(
    "Saved:",
    OUT
    / "p20b_transfer_ablation_fold_results.csv"
)

print(
    "\nP20B COMPLETE"
)
