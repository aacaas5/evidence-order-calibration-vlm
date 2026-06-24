import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = Path("results/p6a/hidden_features.npz")
META = Path("results/p6a/hidden_features_meta.json")

OUT = Path("results/p6b")
OUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load P6A data
# ------------------------------------------------------------

X = np.load(FEATURES)["X"]

meta = json.loads(
    META.read_text(encoding="utf-8")
)

df = pd.DataFrame(meta)

y = df["correct"].astype(int).to_numpy()
groups = df["question_id"].astype(str).to_numpy()

print("Feature matrix:", X.shape)
print("Labels:", y.shape)
print("Correct:", int(y.sum()))
print("Incorrect:", int((1 - y).sum()))


# ------------------------------------------------------------
# Split by QUESTION ID, never by individual condition
# ------------------------------------------------------------

qids = np.array(sorted(set(groups)))

# Trajectory-level stratification:
# 1 = trajectory contains at least one incorrect answer
traj_label = np.array([
    int(np.any(y[groups == qid] == 0))
    for qid in qids
])

train_q, temp_q, train_t, temp_t = train_test_split(
    qids,
    traj_label,
    test_size=0.30,
    random_state=42,
    stratify=traj_label
)

val_q, test_q = train_test_split(
    temp_q,
    test_size=0.50,
    random_state=42,
    stratify=temp_t
)

train_mask = np.isin(groups, train_q)
val_mask = np.isin(groups, val_q)
test_mask = np.isin(groups, test_q)

print("\nQuestion split:")
print("Train:", len(train_q))
print("Val:", len(val_q))
print("Test:", len(test_q))

print("\nCondition split:")
print("Train:", train_mask.sum())
print("Val:", val_mask.sum())
print("Test:", test_mask.sum())


# ------------------------------------------------------------
# Feature sets
# ------------------------------------------------------------

feature_sets = {
    "confidence_only": X[:, -2:-1],
    "confidence_entropy": X[:, -2:],
    "hidden_plus_signals": X,
}


def evaluate(name, features):

    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "probe",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
            )
        )
    ])

    model.fit(
        features[train_mask],
        y[train_mask]
    )

    p = model.predict_proba(
        features[test_mask]
    )[:, 1]

    pred = (p >= 0.5).astype(int)
    yt = y[test_mask]

    metrics = {
        "model": name,
        "test_n": int(len(yt)),
        "accuracy": float(
            accuracy_score(yt, pred)
        ),
        "auroc": float(
            roc_auc_score(yt, p)
        ),
        "auprc": float(
            average_precision_score(yt, p)
        ),
        "brier": float(
            brier_score_loss(yt, p)
        ),
    }

    joblib.dump(
        model,
        OUT / f"{name}.joblib"
    )

    return metrics, p


# ------------------------------------------------------------
# Train + evaluate baselines
# ------------------------------------------------------------

all_metrics = []
test_predictions = df[test_mask].copy()

for name, features in feature_sets.items():

    metrics, probs = evaluate(
        name,
        features
    )

    all_metrics.append(metrics)

    test_predictions[
        f"reliability_{name}"
    ] = probs


metrics_df = pd.DataFrame(all_metrics)

metrics_df.to_csv(
    OUT / "metrics.csv",
    index=False
)

test_predictions.to_csv(
    OUT / "test_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# Save split for reproducibility
# ------------------------------------------------------------

split = {
    "train_question_ids": train_q.tolist(),
    "val_question_ids": val_q.tolist(),
    "test_question_ids": test_q.tolist(),
}

(OUT / "split.json").write_text(
    json.dumps(split, indent=2),
    encoding="utf-8"
)


print("\n" + "=" * 72)
print("P6B BCE RELIABILITY BASELINES")
print("=" * 72)

print(
    metrics_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nSaved:", OUT)
print("\nP6B COMPLETE")
