import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MU = 0.3
MARGIN = 0.05
L2 = 1.0
EPOCHS = 500
LR = 0.01

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

OUT = Path("results/scaled/p19")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load data
# ============================================================

def load_npz(path):
    z = np.load(path)

    if "X" in z:
        return z["X"]

    return z[z.files[0]]


X_mask = load_npz(MASK_X_PATH)
X_blur = load_npz(BLUR_X_PATH)

meta_mask = pd.DataFrame(
    json.loads(
        MASK_META_PATH.read_text(
            encoding="utf-8"
        )
    )
)

meta_blur = pd.DataFrame(
    json.loads(
        BLUR_META_PATH.read_text(
            encoding="utf-8"
        )
    )
)

for df in (meta_mask, meta_blur):

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


# Sort both identically by question/severity
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

X_mask = X_mask[mask_order]
meta_mask = (
    meta_mask.iloc[mask_order]
    .reset_index(drop=True)
)

X_blur = X_blur[blur_order]
meta_blur = (
    meta_blur.iloc[blur_order]
    .reset_index(drop=True)
)


mask_qids = set(
    meta_mask["question_id"]
)

blur_qids = set(
    meta_blur["question_id"]
)

common_qids = sorted(
    mask_qids & blur_qids
)

print("Mask conditions:", len(meta_mask))
print("Blur conditions:", len(meta_blur))
print("Common trajectories:", len(common_qids))
print("Mask feature dim:", X_mask.shape)
print("Blur feature dim:", X_blur.shape)


# ============================================================
# Torch model
# ============================================================

class Probe(nn.Module):

    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_probe(
    X,
    y,
    qids,
    severities,
    use_order=False,
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

    # Balanced BCE weights
    n_pos = max(
        int(y.sum()),
        1
    )

    n_neg = max(
        len(y) - n_pos,
        1
    )

    w_pos = len(y) / (2 * n_pos)
    w_neg = len(y) / (2 * n_neg)

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

    # trajectory index map
    trajectory_groups = {}

    for i, q in enumerate(qids):

        trajectory_groups.setdefault(
            str(q),
            []
        ).append(i)


    best_state = None
    best_loss = float("inf")

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        logits = model(Xt)

        bce_each = (
            nn.functional
            .binary_cross_entropy_with_logits(
                logits,
                yt,
                reduction="none"
            )
        )

        bce = (
            bce_each * wt
        ).mean()


        # Explicit L2
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


        # ----------------------------------------
        # Ordinal loss on MASK training data only
        # ----------------------------------------

        if use_order:

            order_terms = []

            for q, idxs in trajectory_groups.items():

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

                order_loss = torch.stack(
                    order_terms
                ).mean()

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
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }


    model.load_state_dict(
        best_state
    )

    return model


def predict(model, X):

    device = next(
        model.parameters()
    ).device

    Xt = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        logits = model(Xt)

        probs = torch.sigmoid(
            logits
        )

    return probs.cpu().numpy()


# ============================================================
# Metrics
# ============================================================

def emvr_metrics(
    probs,
    df
):

    violations = 0
    pairs = 0
    bad_trajectories = 0
    trajectories = 0

    tmp = df.copy()
    tmp["score"] = probs

    for qid, g in tmp.groupby(
        "question_id"
    ):

        g = g.sort_values(
            "severity"
        )

        if len(g) != 5:
            continue

        trajectories += 1
        vals = g["score"].values

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
        bad_trajectories / trajectories
    )


def basic_metrics(
    y,
    probs
):

    pred = (
        probs >= 0.5
    ).astype(int)

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
    }


def risk_curve(
    y,
    score
):

    order = np.argsort(
        -score
    )

    y = np.asarray(y)[order]

    errors = (
        1 - y
    ).astype(float)

    cumulative_errors = np.cumsum(
        errors
    )

    n = len(y)

    coverage = (
        np.arange(1, n + 1)
        / n
    )

    risk = (
        cumulative_errors
        / np.arange(1, n + 1)
    )

    aurc = np.trapezoid(
        risk,
        coverage
    )

    return coverage, risk, aurc


# ============================================================
# Grouped MASK-train -> BLUR-test
# ============================================================

groups = np.array(
    common_qids
)

gkf = GroupKFold(
    n_splits=5
)

fold_rows = []
oof_rows = []

for fold, (
    train_q_idx,
    test_q_idx
) in enumerate(
    gkf.split(
        groups,
        groups=groups
    ),
    1
):

    train_q = set(
        groups[train_q_idx]
    )

    test_q = set(
        groups[test_q_idx]
    )


    mask_train_idx = (
        meta_mask["question_id"]
        .isin(train_q)
        .values
    )

    blur_test_idx = (
        meta_blur["question_id"]
        .isin(test_q)
        .values
    )


    Xm_train = X_mask[
        mask_train_idx
    ]

    ym_train = (
        meta_mask.loc[
            mask_train_idx,
            "correct"
        ]
        .values
        .astype(int)
    )

    train_meta = (
        meta_mask.loc[
            mask_train_idx
        ]
        .reset_index(drop=True)
    )


    Xb_test = X_blur[
        blur_test_idx
    ]

    yb_test = (
        meta_blur.loc[
            blur_test_idx,
            "correct"
        ]
        .values
        .astype(int)
    )

    test_meta = (
        meta_blur.loc[
            blur_test_idx
        ]
        .reset_index(drop=True)
    )


    # ----------------------------------------
    # scaler learned ONLY from masking train
    # ----------------------------------------

    scaler = StandardScaler()

    Xm_train_s = scaler.fit_transform(
        Xm_train
    )

    Xb_test_s = scaler.transform(
        Xb_test
    )


    # ----------------------------------------
    # BCE
    # ----------------------------------------

    bce_model = train_probe(
        Xm_train_s,
        ym_train,
        train_meta[
            "question_id"
        ].values,
        train_meta[
            "severity"
        ].values,
        use_order=False,
    )

    bce_prob = predict(
        bce_model,
        Xb_test_s
    )


    # ----------------------------------------
    # BCE + order
    # ----------------------------------------

    order_model = train_probe(
        Xm_train_s,
        ym_train,
        train_meta[
            "question_id"
        ].values,
        train_meta[
            "severity"
        ].values,
        use_order=True,
    )

    order_prob = predict(
        order_model,
        Xb_test_s
    )


    # native log-confidence:
    # larger / less-negative = higher confidence
    native_prob = (
        test_meta["c_seq"]
        .values
        .astype(float)
    )


    for name, probs in [
        ("bce_only", bce_prob),
        ("bce_plus_order", order_prob),
    ]:

        m = basic_metrics(
            yb_test,
            probs
        )

        emvr, tvr = emvr_metrics(
            probs,
            test_meta
        )

        fold_rows.append({
            "fold": fold,
            "model": name,
            "test_questions":
                len(test_q),
            "test_conditions":
                len(yb_test),
            **m,
            "emvr": emvr,
            "trajectory_violation_rate":
                tvr,
        })


    # Native metrics
    native_auc_score = native_prob

    native_01 = (
        native_auc_score
        - native_auc_score.min()
    )

    denom = (
        native_01.max()
        + 1e-12
    )

    native_01 = (
        native_01 / denom
    )

    native_m = {
        "accuracy": np.nan,
        "auroc":
            roc_auc_score(
                yb_test,
                native_auc_score
            ),
        "auprc":
            average_precision_score(
                yb_test,
                native_auc_score
            ),
        "brier":
            np.nan,
    }

    n_emvr, n_tvr = emvr_metrics(
        native_auc_score,
        test_meta
    )

    fold_rows.append({
        "fold": fold,
        "model": "native_confidence",
        "test_questions":
            len(test_q),
        "test_conditions":
            len(yb_test),
        **native_m,
        "emvr": n_emvr,
        "trajectory_violation_rate":
            n_tvr,
    })


    # ----------------------------------------
    # Store OOF
    # ----------------------------------------

    for i in range(
        len(test_meta)
    ):

        oof_rows.append({
            "fold": fold,
            "question_id":
                str(
                    test_meta.loc[
                        i,
                        "question_id"
                    ]
                ),
            "severity":
                float(
                    test_meta.loc[
                        i,
                        "severity"
                    ]
                ),
            "correct":
                int(yb_test[i]),
            "native_confidence":
                float(native_prob[i]),
            "bce_only":
                float(bce_prob[i]),
            "bce_plus_order":
                float(order_prob[i]),
        })


    print(
        f"Fold {fold}/5 complete | "
        f"mask train q={len(train_q)} | "
        f"blur test q={len(test_q)}"
    )


# ============================================================
# Summaries
# ============================================================

fold_df = pd.DataFrame(
    fold_rows
)

oof = pd.DataFrame(
    oof_rows
)

summary = (
    fold_df
    .groupby("model")
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
        ),
    )
    .reset_index()
)


# ============================================================
# Selective prediction
# ============================================================

risk_rows = []

plt.figure(
    figsize=(9, 6)
)

for model_name in [
    "native_confidence",
    "bce_only",
    "bce_plus_order",
]:

    coverage, risk, aurc = (
        risk_curve(
            oof["correct"].values,
            oof[model_name].values
        )
    )

    row = {
        "model": model_name,
        "aurc": aurc,
    }

    for target in [
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
    ]:

        idx = np.argmin(
            np.abs(
                coverage - target
            )
        )

        row[
            f"risk_at_{int(target*100)}pct"
        ] = risk[idx]

    risk_rows.append(row)

    plt.plot(
        coverage,
        risk,
        label=model_name
    )


plt.xlabel("Coverage")
plt.ylabel("Risk")
plt.title(
    "P19 Mask-Train → Blur-Test Risk-Coverage"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "blur_risk_coverage.png",
    dpi=180
)

plt.close()

risk_df = pd.DataFrame(
    risk_rows
)


# ============================================================
# Paired trajectory bootstrap
# ============================================================

rng = np.random.default_rng(
    SEED
)

question_ids = (
    oof["question_id"]
    .unique()
)

boot = []

for _ in range(1000):

    sampled = rng.choice(
        question_ids,
        size=len(question_ids),
        replace=True
    )

    parts = []

    # Preserve duplicate sampled trajectories
    for j, q in enumerate(sampled):

        g = oof[
            oof["question_id"] == q
        ].copy()

        g["boot_id"] = j

        parts.append(g)

    b = pd.concat(
        parts,
        ignore_index=True
    )


    def boot_emvr(col):

        violations = 0
        pairs = 0

        for _, g in b.groupby(
            "boot_id"
        ):

            g = g.sort_values(
                "severity"
            )

            vals = g[col].values

            if len(vals) != 5:
                continue

            for a, c in zip(
                vals[:-1],
                vals[1:]
            ):

                violations += int(
                    c > a
                )

                pairs += 1

        return violations / pairs


    emvr_bce = boot_emvr(
        "bce_only"
    )

    emvr_order = boot_emvr(
        "bce_plus_order"
    )


    y = b[
        "correct"
    ].values

    p_bce = b[
        "bce_only"
    ].values

    p_order = b[
        "bce_plus_order"
    ].values


    try:
        auc_bce = roc_auc_score(
            y,
            p_bce
        )

        auc_order = roc_auc_score(
            y,
            p_order
        )

    except ValueError:
        continue


    _, _, aurc_bce = risk_curve(
        y,
        p_bce
    )

    _, _, aurc_order = risk_curve(
        y,
        p_order
    )


    brier_bce = brier_score_loss(
        y,
        p_bce
    )

    brier_order = brier_score_loss(
        y,
        p_order
    )


    boot.append({
        "delta_emvr":
            emvr_order - emvr_bce,

        "delta_auroc":
            auc_order - auc_bce,

        "delta_aurc":
            aurc_order - aurc_bce,

        "delta_brier":
            brier_order - brier_bce,
    })


boot_df = pd.DataFrame(
    boot
)


def ci(col):

    vals = boot_df[col].dropna()

    return {
        "mean":
            float(vals.mean()),

        "lower_95":
            float(
                np.percentile(
                    vals,
                    2.5
                )
            ),

        "upper_95":
            float(
                np.percentile(
                    vals,
                    97.5
                )
            ),
    }


bootstrap_summary = {
    "seed": SEED,
    "resamples": len(
        boot_df
    ),
    "order_minus_bce": {
        "emvr":
            ci("delta_emvr"),

        "auroc":
            ci("delta_auroc"),

        "aurc":
            ci("delta_aurc"),

        "brier":
            ci("delta_brier"),
    }
}


# ============================================================
# Save
# ============================================================

fold_df.to_csv(
    OUT / "transfer_fold_results.csv",
    index=False
)

summary.to_csv(
    OUT / "transfer_summary.csv",
    index=False
)

oof.to_csv(
    OUT / "transfer_oof_predictions.csv",
    index=False
)

risk_df.to_csv(
    OUT / "blur_risk_coverage.csv",
    index=False
)

boot_df.to_csv(
    OUT / "bootstrap_transfer_samples.csv",
    index=False
)

(
    OUT / "bootstrap_transfer.json"
).write_text(
    json.dumps(
        bootstrap_summary,
        indent=2
    ),
    encoding="utf-8"
)


# ============================================================
# Final print
# ============================================================

print("\n" + "=" * 100)
print("P19 HELD-OUT BLUR TRANSFER")
print("=" * 100)

print("\nTRANSFER SUMMARY")

print(
    summary.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)

print("\nSELECTIVE PREDICTION")

print(
    risk_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)

print("\nORDER MINUS BCE BOOTSTRAP")

for metric, values in (
    bootstrap_summary[
        "order_minus_bce"
    ].items()
):

    print(
        f"{metric}: "
        f"mean={values['mean']:.4f} "
        f"95% CI=["
        f"{values['lower_95']:.4f}, "
        f"{values['upper_95']:.4f}]"
    )

print("\nSaved:", OUT)
print("P19D COMPLETE")
print("P19 COMPLETE")
