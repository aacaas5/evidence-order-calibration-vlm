import json
from pathlib import Path

import pandas as pd

ROOT = Path("results/scaled")
OUT = ROOT / "p20"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load existing results
# ============================================================

p20a = pd.read_csv(
    OUT / "p20a_ablation_summary.csv"
)

p20b = pd.read_csv(
    OUT / "p20b_transfer_ablation_summary.csv"
)

transfer = pd.read_csv(
    ROOT / "p19" / "transfer_summary.csv"
)

risk = pd.read_csv(
    ROOT / "p19" / "blur_risk_coverage.csv"
)

bootstrap_transfer = json.loads(
    (
        ROOT
        / "p19"
        / "bootstrap_transfer.json"
    ).read_text(
        encoding="utf-8"
    )
)

blur_summary = json.loads(
    (
        ROOT
        / "p19"
        / "blur_summary.json"
    ).read_text(
        encoding="utf-8"
    )
)

# ============================================================
# Primary frozen masking results
# ============================================================

primary_masking = pd.DataFrame([
    {
        "setting": "masking",
        "model": "native_confidence",
        "accuracy": None,
        "auroc": 0.76885,
        "auprc": None,
        "brier": None,
        "emvr": 0.43608,
        "trajectory_violation_rate": None,
        "aurc": 0.265415,
        "status": "primary_frozen"
    },
    {
        "setting": "masking",
        "model": "bce_only",
        "accuracy": 0.61044,
        "auroc": 0.674279,
        "auprc": 0.723686,
        "brier": 0.226836,
        "emvr": 0.329841,
        "trajectory_violation_rate": 0.812698,
        "aurc": 0.315096,
        "status": "primary_frozen"
    },
    {
        "setting": "masking",
        "model": "bce_plus_order",
        "accuracy": 0.603556,
        "auroc": 0.676207,
        "auprc": 0.722245,
        "brier": 0.225634,
        "emvr": 0.302698,
        "trajectory_violation_rate": 0.767302,
        "aurc": 0.315798,
        "status": "primary_frozen"
    }
])

# ============================================================
# Primary held-out blur transfer
# ============================================================

primary_transfer = transfer.copy()

primary_transfer["setting"] = (
    "mask_train_blur_test"
)

primary_transfer["status"] = (
    "primary_frozen"
)

# ============================================================
# P19 blur degradation sanity summary
# ============================================================

severity_rows = []

for sev, values in (
    blur_summary[
        "severity_summary"
    ].items()
):

    severity_rows.append({
        "severity": float(sev),
        "n": values["n"],
        "accuracy": values["accuracy"],
        "mean_c_seq": values["mean_c_seq"],
        "mean_entropy": values["mean_entropy"]
    })

blur_severity = pd.DataFrame(
    severity_rows
)

blur_global = pd.DataFrame([
    {
        "trajectories":
            blur_summary[
                "trajectories"
            ],
        "conditions":
            blur_summary[
                "conditions"
            ],
        "emvr":
            blur_summary[
                "emvr"
            ],
        "trajectory_violation_rate":
            blur_summary[
                "trajectory_violation_rate"
            ],
        "clean_correct_trajectories":
            blur_summary[
                "clean_correct_trajectories"
            ],
        "clean_correct_emvr":
            blur_summary[
                "clean_correct_emvr"
            ]
    }
])

# ============================================================
# Bootstrap transfer summary
# ============================================================

boot_rows = []

for metric, values in (
    bootstrap_transfer[
        "order_minus_bce"
    ].items()
):

    boot_rows.append({
        "setting":
            "mask_train_blur_test",
        "comparison":
            "bce_plus_order_minus_bce_only",
        "metric":
            metric,
        "mean":
            values["mean"],
        "lower_95":
            values["lower_95"],
        "upper_95":
            values["upper_95"]
    })

boot_transfer_df = pd.DataFrame(
    boot_rows
)

# ============================================================
# Frozen masking bootstrap values
# ============================================================

mask_bootstrap = pd.DataFrame([
    {
        "setting": "masking",
        "comparison":
            "bce_plus_order_minus_bce_only",
        "metric": "emvr",
        "mean": -0.026989,
        "lower_95": -0.044034,
        "upper_95": -0.009943
    },
    {
        "setting": "masking",
        "comparison":
            "bce_plus_order_minus_bce_only",
        "metric": "auroc",
        "mean": 0.003773,
        "lower_95": -0.002943,
        "upper_95": 0.010189
    },
    {
        "setting": "masking",
        "comparison":
            "bce_plus_order_minus_bce_only",
        "metric": "aurc",
        "mean": 0.000702,
        "lower_95": -0.004039,
        "upper_95": 0.007011
    }
])

bootstrap_all = pd.concat(
    [
        mask_bootstrap,
        boot_transfer_df
    ],
    ignore_index=True
)

# ============================================================
# Critical vs matched non-critical control
# ============================================================

control_table = pd.DataFrame([
    {
        "metric":
            "accuracy_drop_difference",
        "comparison":
            "critical_minus_noncritical",
        "mean":
            0.276471,
        "lower_95":
            0.200000,
        "upper_95":
            0.347206
    },
    {
        "metric":
            "confidence_change_difference",
        "comparison":
            "critical_minus_noncritical",
        "mean":
            -0.252755,
        "lower_95":
            -0.321500,
        "upper_95":
            -0.189350
    },
    {
        "metric":
            "entropy_change_difference",
        "comparison":
            "critical_minus_noncritical",
        "mean":
            0.542872,
        "lower_95":
            0.423380,
        "upper_95":
            0.667770
    }
])

# ============================================================
# Paper claims map
# ============================================================

claims = [
    {
        "claim_id": "C1",
        "claim":
            "Progressive removal of question-critical evidence "
            "reduces VLM task performance.",
        "evidence":
            "Critical masking and blur severity curves.",
        "strength":
            "strong"
    },
    {
        "claim_id": "C2",
        "claim":
            "Native VLM confidence responds to evidence loss "
            "in aggregate but is frequently non-monotonic "
            "within individual evidence trajectories.",
        "evidence":
            "Masking EMVR and blur EMVR.",
        "strength":
            "strong"
    },
    {
        "claim_id": "C3",
        "claim":
            "Question-critical masking causes substantially "
            "larger degradation than matched non-critical masking.",
        "evidence":
            "P18 paired bootstrap control.",
        "strength":
            "strong"
    },
    {
        "claim_id": "C4",
        "claim":
            "Evidence-order regularization reduces evidence-order "
            "violations compared with BCE-only training.",
        "evidence":
            "Primary masking bootstrap.",
        "strength":
            "strong"
    },
    {
        "claim_id": "C5",
        "claim":
            "The evidence-order benefit transfers from masking "
            "to an unseen local blur degradation.",
        "evidence":
            "P19 paired blur-transfer bootstrap.",
        "strength":
            "strong"
    },
    {
        "claim_id": "C6",
        "claim":
            "The method improves selective prediction over "
            "native confidence.",
        "evidence":
            "Not supported.",
        "strength":
            "do_not_claim"
    },
    {
        "claim_id": "C7",
        "claim":
            "The method significantly improves AUROC over BCE-only.",
        "evidence":
            "Bootstrap CIs cross zero.",
        "strength":
            "do_not_claim"
    }
]

claims_df = pd.DataFrame(
    claims
)

# ============================================================
# Save
# ============================================================

primary_masking.to_csv(
    OUT / "paper_primary_masking.csv",
    index=False
)

primary_transfer.to_csv(
    OUT / "paper_primary_transfer.csv",
    index=False
)

p20a.to_csv(
    OUT / "paper_ablation_masking.csv",
    index=False
)

p20b.to_csv(
    OUT / "paper_ablation_transfer.csv",
    index=False
)

bootstrap_all.to_csv(
    OUT / "paper_bootstrap_summary.csv",
    index=False
)

control_table.to_csv(
    OUT / "paper_control_summary.csv",
    index=False
)

blur_severity.to_csv(
    OUT / "paper_blur_severity.csv",
    index=False
)

blur_global.to_csv(
    OUT / "paper_blur_global.csv",
    index=False
)

claims_df.to_csv(
    OUT / "paper_claims_map.csv",
    index=False
)

risk.to_csv(
    OUT / "paper_blur_risk_coverage.csv",
    index=False
)

# ============================================================
# Compact JSON handoff for Codex later
# ============================================================

handoff = {
    "primary_claims": claims,
    "primary_masking":
        primary_masking.to_dict(
            orient="records"
        ),
    "primary_transfer":
        primary_transfer.to_dict(
            orient="records"
        ),
    "bootstrap":
        bootstrap_all.to_dict(
            orient="records"
        ),
    "control":
        control_table.to_dict(
            orient="records"
        ),
    "important_rules": [
        "Do not claim selective prediction improvement over native confidence.",
        "Do not claim statistically significant AUROC improvement.",
        "Use frozen primary results for main claims.",
        "Use P20A and P20B only as ablations.",
        "Describe matched irrelevant regions as matched non-critical regions.",
        "Describe audit as automated visual quality control with limited human spot-check.",
        "Qwen remains frozen in all experiments."
    ]
}

(
    OUT
    / "paper_results_handoff.json"
).write_text(
    json.dumps(
        handoff,
        indent=2
    ),
    encoding="utf-8"
)

# ============================================================
# Print
# ============================================================

print("\n" + "=" * 100)
print("P20C PAPER-READY CONSOLIDATION")
print("=" * 100)

print("\nPRIMARY MASKING")
print(
    primary_masking.to_string(
        index=False
    )
)

print("\nPRIMARY TRANSFER")
print(
    primary_transfer.to_string(
        index=False
    )
)

print("\nBOOTSTRAP")
print(
    bootstrap_all.to_string(
        index=False
    )
)

print("\nCLAIMS")
print(
    claims_df[
        [
            "claim_id",
            "strength",
            "claim"
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved paper-ready files in:")
print(OUT)

print("\nP20C COMPLETE")
print("P20 COMPLETE")
