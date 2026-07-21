import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 42
N_BOOTSTRAP = 1000
SEVERITIES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
P5_RESULTS = Path("results/scaled/p5/results.json")
P18_RESULTS = Path("results/scaled/p18/irrelevant_results.json")
CONTROL_BOXES = Path("results/scaled/p18/control_boxes.json")
OUT_DIR = Path("results/scaled/p18")
METRICS_PATH = OUT_DIR / "control_metrics.json"
BOOTSTRAP_PATH = OUT_DIR / "bootstrap_control.json"
PAIRED_PATH = OUT_DIR / "paired_effect_summary.csv"


def load_rows(path):
    frame = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    frame["question_id"] = frame["question_id"].astype(str)
    frame["severity"] = frame["severity"].astype(float)
    frame["correct"] = frame["correct"].astype(bool)
    return frame


def trajectory_arrays(frame, question_ids):
    arrays = {}
    for question_id in question_ids:
        trajectory = frame[frame["question_id"] == question_id].sort_values("severity")
        if len(trajectory) != len(SEVERITIES) or not np.allclose(trajectory["severity"], SEVERITIES):
            raise RuntimeError(f"Incomplete trajectory in {question_id}")
        arrays[question_id] = {
            "correct": trajectory["correct"].to_numpy(bool),
            "c_seq": trajectory["c_seq"].to_numpy(float),
            "entropy": trajectory["entropy"].to_numpy(float),
            "category": str(trajectory.iloc[0]["category"]),
        }
    return arrays


def subset_metrics(arrays, question_ids):
    if not question_ids:
        return {
            "trajectory_count": 0,
            "condition_count": 0,
            "severity_summary": {},
            "adjacent_violations": 0,
            "adjacent_pairs": 0,
            "emvr": None,
            "trajectories_with_violation": 0,
            "trajectory_violation_rate": None,
            "clean_correct_trajectories": 0,
            "clean_correct_emvr": None,
        }
    correct = np.stack([arrays[q]["correct"] for q in question_ids])
    confidence = np.stack([arrays[q]["c_seq"] for q in question_ids])
    entropy = np.stack([arrays[q]["entropy"] for q in question_ids])
    violations = confidence[:, 1:] > confidence[:, :-1]
    clean_correct = correct[:, 0]
    clean_violations = violations[clean_correct]
    return {
        "trajectory_count": len(question_ids),
        "condition_count": len(question_ids) * len(SEVERITIES),
        "severity_summary": {
            str(float(severity)): {
                "n": len(question_ids),
                "accuracy": float(correct[:, index].mean()),
                "mean_c_seq": float(confidence[:, index].mean()),
                "mean_entropy": float(entropy[:, index].mean()),
            }
            for index, severity in enumerate(SEVERITIES)
        },
        "adjacent_violations": int(violations.sum()),
        "adjacent_pairs": int(violations.size),
        "emvr": float(violations.mean()),
        "trajectories_with_violation": int(violations.any(axis=1).sum()),
        "trajectory_violation_rate": float(violations.any(axis=1).mean()),
        "clean_correct_trajectories": int(clean_correct.sum()),
        "clean_correct_emvr": float(clean_violations.mean()) if clean_violations.size else None,
        "changes_full_minus_clean": {
            "accuracy": float(correct[:, -1].mean() - correct[:, 0].mean()),
            "mean_c_seq": float(confidence[:, -1].mean() - confidence[:, 0].mean()),
            "mean_entropy": float(entropy[:, -1].mean() - entropy[:, 0].mean()),
        },
    }


def intervention_metrics(arrays, question_ids):
    overall = subset_metrics(arrays, question_ids)
    categories = sorted({arrays[q]["category"] for q in question_ids})
    overall["by_category"] = {
        category: subset_metrics(
            arrays, [q for q in question_ids if arrays[q]["category"] == category]
        )
        for category in categories
    }
    return overall


def interval(values):
    values = np.asarray(values, dtype=float)
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "bootstrap_mean": float(values.mean()),
        "valid_resamples": int(len(values)),
    }


controls = json.loads(CONTROL_BOXES.read_text(encoding="utf-8"))
valid_ids = sorted(str(row["question_id"]) for row in controls if row["control_valid"])
invalid_count = sum(not row["control_valid"] for row in controls)
critical_frame = load_rows(P5_RESULTS)
irrelevant_frame = load_rows(P18_RESULTS)
critical = trajectory_arrays(critical_frame, valid_ids)
irrelevant = trajectory_arrays(irrelevant_frame, valid_ids)

critical_metrics = intervention_metrics(critical, valid_ids)
irrelevant_metrics = intervention_metrics(irrelevant, valid_ids)

paired_rows = []
for question_id in valid_ids:
    critical_row = critical[question_id]
    irrelevant_row = irrelevant[question_id]
    critical_violations = int(np.sum(critical_row["c_seq"][1:] > critical_row["c_seq"][:-1]))
    irrelevant_violations = int(np.sum(irrelevant_row["c_seq"][1:] > irrelevant_row["c_seq"][:-1]))
    critical_loss = int(critical_row["correct"][0]) - int(critical_row["correct"][-1])
    irrelevant_loss = int(irrelevant_row["correct"][0]) - int(irrelevant_row["correct"][-1])
    critical_confidence_change = critical_row["c_seq"][-1] - critical_row["c_seq"][0]
    irrelevant_confidence_change = irrelevant_row["c_seq"][-1] - irrelevant_row["c_seq"][0]
    critical_entropy_change = critical_row["entropy"][-1] - critical_row["entropy"][0]
    irrelevant_entropy_change = irrelevant_row["entropy"][-1] - irrelevant_row["entropy"][0]
    paired_rows.append({
        "question_id": question_id,
        "category": critical_row["category"],
        "critical_adjacent_violations": critical_violations,
        "irrelevant_adjacent_violations": irrelevant_violations,
        "critical_minus_irrelevant_adjacent_violations": critical_violations - irrelevant_violations,
        "critical_correctness_loss": critical_loss,
        "irrelevant_correctness_loss": irrelevant_loss,
        "critical_minus_irrelevant_accuracy_drop": critical_loss - irrelevant_loss,
        "critical_confidence_change": critical_confidence_change,
        "irrelevant_confidence_change": irrelevant_confidence_change,
        "critical_minus_irrelevant_confidence_change": critical_confidence_change - irrelevant_confidence_change,
        "critical_entropy_change": critical_entropy_change,
        "irrelevant_entropy_change": irrelevant_entropy_change,
        "critical_minus_irrelevant_entropy_change": critical_entropy_change - irrelevant_entropy_change,
    })
paired = pd.DataFrame(paired_rows)
paired.to_csv(PAIRED_PATH, index=False)

point_estimates = {
    "critical_minus_irrelevant_accuracy_drop": float(paired["critical_minus_irrelevant_accuracy_drop"].mean()),
    "critical_minus_irrelevant_confidence_change": float(paired["critical_minus_irrelevant_confidence_change"].mean()),
    "critical_minus_irrelevant_entropy_change": float(paired["critical_minus_irrelevant_entropy_change"].mean()),
    "critical_minus_irrelevant_emvr": float(
        paired["critical_minus_irrelevant_adjacent_violations"].mean() / 4
    ),
}
rng = np.random.default_rng(SEED)
bootstrap_values = {name: [] for name in point_estimates}
for iteration in range(N_BOOTSTRAP):
    indices = rng.integers(0, len(paired), size=len(paired))
    sample = paired.iloc[indices]
    bootstrap_values["critical_minus_irrelevant_accuracy_drop"].append(
        sample["critical_minus_irrelevant_accuracy_drop"].mean()
    )
    bootstrap_values["critical_minus_irrelevant_confidence_change"].append(
        sample["critical_minus_irrelevant_confidence_change"].mean()
    )
    bootstrap_values["critical_minus_irrelevant_entropy_change"].append(
        sample["critical_minus_irrelevant_entropy_change"].mean()
    )
    bootstrap_values["critical_minus_irrelevant_emvr"].append(
        sample["critical_minus_irrelevant_adjacent_violations"].mean() / 4
    )

bootstrap_summary = {
    "seed": SEED,
    "bootstrap_resamples": N_BOOTSTRAP,
    "resampling_unit": "question_trajectory",
    "difference_direction": "critical minus matched irrelevant",
    "accuracy_drop_definition": "clean correctness minus full-mask correctness",
    "metrics": {
        name: {"point_estimate": point_estimates[name], **interval(bootstrap_values[name])}
        for name in point_estimates
    },
}
BOOTSTRAP_PATH.write_text(json.dumps(bootstrap_summary, indent=2), encoding="utf-8")

summary = {
    "valid_trajectories": len(valid_ids),
    "invalid_trajectories": invalid_count,
    "critical": critical_metrics,
    "irrelevant": irrelevant_metrics,
    "paired_difference": bootstrap_summary,
}
METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

plot_specs = [
    ("accuracy", "accuracy", "Accuracy", "accuracy_critical_vs_irrelevant.png"),
    ("mean_c_seq", "mean_c_seq", "Mean sequence confidence (c_seq)", "confidence_critical_vs_irrelevant.png"),
    ("mean_entropy", "mean_entropy", "Mean token entropy", "entropy_critical_vs_irrelevant.png"),
]
for _, field, ylabel, filename in plot_specs:
    plt.figure(figsize=(7.2, 4.8))
    for label, metrics, color, marker in (
        ("Critical-region mask", critical_metrics, "#b2182b", "o"),
        ("Matched irrelevant-region mask", irrelevant_metrics, "#2166ac", "s"),
    ):
        values = [metrics["severity_summary"][str(float(s))][field] for s in SEVERITIES]
        plt.plot(SEVERITIES, values, label=label, color=color, marker=marker, linewidth=2)
    plt.xlabel("Mask severity (lambda)")
    plt.ylabel(ylabel)
    plt.xticks(SEVERITIES)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=180)
    plt.close()

def fmt(value):
    return f"{value:.6f}"


critical_clean = critical_metrics["severity_summary"]["0.0"]
critical_full = critical_metrics["severity_summary"]["1.0"]
irrelevant_clean = irrelevant_metrics["severity_summary"]["0.0"]
irrelevant_full = irrelevant_metrics["severity_summary"]["1.0"]
print("P18 MATCHED IRRELEVANT CONTROL")
print()
print("Valid trajectories:", len(valid_ids))
print("Invalid trajectories:", invalid_count)
print()
print("CRITICAL")
print("clean accuracy:", fmt(critical_clean["accuracy"]))
print("full-mask accuracy:", fmt(critical_full["accuracy"]))
print("accuracy drop:", fmt(critical_clean["accuracy"] - critical_full["accuracy"]))
print("EMVR:", fmt(critical_metrics["emvr"]))
print("clean-correct EMVR:", fmt(critical_metrics["clean_correct_emvr"]))
print()
print("IRRELEVANT")
print("clean accuracy:", fmt(irrelevant_clean["accuracy"]))
print("full-mask accuracy:", fmt(irrelevant_full["accuracy"]))
print("accuracy drop:", fmt(irrelevant_clean["accuracy"] - irrelevant_full["accuracy"]))
print("EMVR:", fmt(irrelevant_metrics["emvr"]))
print("clean-correct EMVR:", fmt(irrelevant_metrics["clean_correct_emvr"]))
print()
print("PAIRED DIFFERENCE")
for label, name in (
    ("critical-minus-irrelevant accuracy drop", "critical_minus_irrelevant_accuracy_drop"),
    ("critical-minus-irrelevant EMVR", "critical_minus_irrelevant_emvr"),
    ("critical-minus-irrelevant confidence change", "critical_minus_irrelevant_confidence_change"),
    ("critical-minus-irrelevant entropy change", "critical_minus_irrelevant_entropy_change"),
):
    result = bootstrap_summary["metrics"][name]
    print(label + ":", fmt(result["point_estimate"]))
    print("95% CI:", f"[{fmt(result['lower_95'])}, {fmt(result['upper_95'])}]")
    print()
print("Output directory:", OUT_DIR)
