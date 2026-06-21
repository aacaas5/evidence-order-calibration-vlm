import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt

IN = Path("results/p5b/results.json")
OUT = Path("results/p5c")
OUT.mkdir(parents=True, exist_ok=True)

rows = json.loads(IN.read_text(encoding="utf-8"))
df = pd.DataFrame(rows)

# ------------------------------------------------------------
# Per-severity summary
# ------------------------------------------------------------

summary = (
    df.groupby("severity")
      .agg(
          n=("question_id", "count"),
          accuracy=("correct", "mean"),
          mean_c_seq=("c_seq", "mean"),
          mean_entropy=("entropy", "mean"),
      )
      .reset_index()
)

summary.to_csv(OUT / "severity_summary.csv", index=False)

print("\nALL TRAJECTORIES")
print(summary.to_string(index=False))


# ------------------------------------------------------------
# Clean-correct subset
# ------------------------------------------------------------

clean_correct_ids = set(
    df[
        (df["severity"] == 0.0)
        & (df["correct"] == True)
    ]["question_id"]
)

clean_df = df[
    df["question_id"].isin(clean_correct_ids)
].copy()

clean_summary = (
    clean_df.groupby("severity")
            .agg(
                n=("question_id", "count"),
                accuracy=("correct", "mean"),
                mean_c_seq=("c_seq", "mean"),
                mean_entropy=("entropy", "mean"),
            )
            .reset_index()
)

clean_summary.to_csv(
    OUT / "clean_correct_severity_summary.csv",
    index=False
)

print("\nCLEAN-CORRECT TRAJECTORIES")
print("Trajectories:", len(clean_correct_ids))
print(clean_summary.to_string(index=False))


# ------------------------------------------------------------
# Per-trajectory violation analysis
# ------------------------------------------------------------

traj_rows = []

for qid, g in df.groupby("question_id"):
    g = g.sort_values("severity")

    if len(g) != 5:
        continue

    conf = g["c_seq"].tolist()

    violations = sum(
        conf[i + 1] > conf[i]
        for i in range(4)
    )

    traj_rows.append({
        "question_id": qid,
        "question": g.iloc[0]["question"],
        "ground_truth": g.iloc[0]["ground_truth"],
        "clean_correct": bool(g.iloc[0]["correct"]),
        "violations": violations,
        "emvr": violations / 4,
        "clean_c_seq": conf[0],
        "full_loss_c_seq": conf[-1],
        "confidence_change": conf[-1] - conf[0],
        "clean_answer": g.iloc[0]["answer"],
        "full_loss_answer": g.iloc[-1]["answer"],
        "full_loss_correct": bool(g.iloc[-1]["correct"]),
    })

traj = pd.DataFrame(traj_rows)

traj.to_csv(
    OUT / "trajectory_analysis.csv",
    index=False
)


def calc_emvr(frame):
    subset = traj[
        traj["question_id"].isin(
            frame["question_id"].unique()
        )
    ]

    total_v = subset["violations"].sum()
    total_pairs = len(subset) * 4

    return (
        total_v / total_pairs
        if total_pairs else float("nan")
    )


all_emvr = calc_emvr(df)
clean_emvr = calc_emvr(clean_df)

print("\nEMVR")
print("All trajectories:", round(all_emvr, 4))
print("Clean-correct only:", round(clean_emvr, 4))


# ------------------------------------------------------------
# Strong failure cases
# ------------------------------------------------------------

failures = traj[
    traj["clean_correct"] == True
].sort_values(
    ["violations", "confidence_change"],
    ascending=[False, False]
)

failures.head(10).to_csv(
    OUT / "top_failure_cases.csv",
    index=False
)

print("\nTOP FAILURE CASES")
print(
    failures[
        [
            "question_id",
            "question",
            "ground_truth",
            "violations",
            "confidence_change",
            "full_loss_answer",
            "full_loss_correct",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# FIGURE 1 - Accuracy
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))
plt.plot(
    summary["severity"],
    summary["accuracy"],
    marker="o"
)
plt.xlabel("Critical evidence removed")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Critical Evidence Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUT / "accuracy_vs_evidence.png",
    dpi=200
)
plt.close()


# ------------------------------------------------------------
# FIGURE 2 - Native confidence
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))
plt.plot(
    summary["severity"],
    summary["mean_c_seq"],
    marker="o"
)
plt.xlabel("Critical evidence removed")
plt.ylabel("Mean sequence log-confidence")
plt.title("Native Confidence vs Evidence Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUT / "confidence_vs_evidence.png",
    dpi=200
)
plt.close()


# ------------------------------------------------------------
# FIGURE 3 - Entropy
# ------------------------------------------------------------

plt.figure(figsize=(6, 4))
plt.plot(
    summary["severity"],
    summary["mean_entropy"],
    marker="o"
)
plt.xlabel("Critical evidence removed")
plt.ylabel("Mean predictive entropy")
plt.title("Entropy vs Critical Evidence Loss")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUT / "entropy_vs_evidence.png",
    dpi=200
)
plt.close()


# ------------------------------------------------------------
# FIGURE 4 - Individual confidence trajectories
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

for qid, g in clean_df.groupby("question_id"):
    g = g.sort_values("severity")

    plt.plot(
        g["severity"],
        g["c_seq"],
        alpha=0.35
    )

plt.xlabel("Critical evidence removed")
plt.ylabel("Sequence log-confidence")
plt.title("Clean-Correct Confidence Trajectories")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUT / "individual_confidence_trajectories.png",
    dpi=200
)
plt.close()


# ------------------------------------------------------------
# Save headline numbers
# ------------------------------------------------------------

headline = {
    "total_trajectories": int(traj.shape[0]),
    "clean_correct_trajectories": int(len(clean_correct_ids)),
    "all_emvr": float(all_emvr),
    "clean_correct_emvr": float(clean_emvr),
    "trajectories_with_violation": int(
        (traj["violations"] > 0).sum()
    ),
    "clean_correct_with_violation": int(
        (
            (traj["clean_correct"] == True)
            & (traj["violations"] > 0)
        ).sum()
    ),
}

(OUT / "headline_results.json").write_text(
    json.dumps(headline, indent=2),
    encoding="utf-8"
)

print("\nHEADLINE RESULTS")
for k, v in headline.items():
    print(f"{k}: {v}")

print("\nSaved analysis to:", OUT)
