"""Validate paper/repository claims against frozen experiment artifacts."""

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/gqa/manifests/gqa_evidence_scaled_accepted.json"
EXPECTED_MANIFEST_SHA256 = "FD991EF4101D105CBB80F256B9CE0001C83DF2529E02C93E3B0B1CE4C11820EA"
failures = []


def require(condition, message):
    if not condition:
        failures.append(message)


def close(actual, expected, label, tolerance=5e-7):
    require(np.isclose(float(actual), float(expected), atol=tolerance, rtol=0),
            f"{label}: expected {expected}, got {actual}")


digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest().upper()
require(digest == EXPECTED_MANIFEST_SHA256, f"accepted manifest hash changed: {digest}")
manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
require(len(manifest) == 176, f"accepted manifest length is {len(manifest)}, expected 176")

p5 = json.loads((ROOT / "results/scaled/p5/summary.json").read_text(encoding="utf-8"))
require(p5["conditions"] == 880, "P5 must contain 880 conditions")
close(p5["emvr"], 0.43607954545454547, "native masking EMVR")
close(p5["trajectory_violation_rate"], 0.9204545454545454, "native trajectory violation rate")
close(p5["severity_summary"]["0.0"]["accuracy"], 0.6420454545454546, "clean accuracy")
close(p5["severity_summary"]["1.0"]["accuracy"], 0.35795454545454547, "full-mask accuracy")

p18 = json.loads((ROOT / "results/scaled/p18/control_metrics.json").read_text(encoding="utf-8"))
require(p18["valid_trajectories"] == 170 and p18["invalid_trajectories"] == 6,
        "P18 valid/invalid counts must be 170/6")
close(p18["paired_difference"]["metrics"]["critical_minus_irrelevant_accuracy_drop"]["point_estimate"],
      0.27647058823529413, "P18 accuracy-drop difference")

p18_rows = json.loads((ROOT / "results/scaled/p18/irrelevant_results.json").read_text(encoding="utf-8"))
require(len(p18_rows) == 850, f"P18 row count is {len(p18_rows)}, expected 850")
require(len({(str(row["question_id"]), float(row["severity"])) for row in p18_rows}) == 850,
        "P18 contains duplicate/missing question-severity keys")

blur_rows = json.loads((ROOT / "results/scaled/p19/blur_results.json").read_text(encoding="utf-8"))
require(len(blur_rows) == 880, f"P19 blur row count is {len(blur_rows)}, expected 880")

mask = pd.read_csv(ROOT / "results/scaled/p20/paper_primary_masking.csv")
transfer = pd.read_csv(ROOT / "results/scaled/p20/paper_primary_transfer.csv")
bootstrap = pd.read_csv(ROOT / "results/scaled/p20/paper_bootstrap_summary.csv")
close(mask.loc[mask.model == "bce_only", "emvr"].iloc[0], 0.329841, "BCE masking EMVR")
close(mask.loc[mask.model == "bce_plus_order", "emvr"].iloc[0], 0.302698, "order masking EMVR")
close(transfer.loc[transfer.model == "bce_only", "emvr_mean"].iloc[0], 0.4492460317460317,
      "BCE transfer EMVR")
close(transfer.loc[transfer.model == "bce_plus_order", "emvr_mean"].iloc[0], 0.4022222222222222,
      "order transfer EMVR")

mask_boot = bootstrap[(bootstrap.setting == "masking") & (bootstrap.metric == "emvr")].iloc[0]
blur_boot = bootstrap[(bootstrap.setting == "mask_train_blur_test") & (bootstrap.metric == "emvr")].iloc[0]
close(mask_boot["mean"], -0.026989, "masking paired EMVR delta")
close(mask_boot["lower_95"], -0.044034, "masking EMVR CI lower")
close(mask_boot["upper_95"], -0.009943, "masking EMVR CI upper")
close(blur_boot["mean"], -0.04677840909090909, "transfer paired EMVR delta")
close(blur_boot["lower_95"], -0.07386363636363635, "transfer EMVR CI lower")
close(blur_boot["upper_95"], -0.019886363636363646, "transfer EMVR CI upper")

required_files = [
    "README.md", "LICENSE", "requirements.txt", "docs/REPRODUCIBILITY.md",
    "docs/EXPERIMENT_MAP.md", "docs/RESULTS_INDEX.md", "docs/SCRIPT_INDEX.md",
    "paper/main.tex", "paper/references.bib", "paper/figures/pipeline.png",
    "paper/figures/critical_mask_severity.png", "paper/figures/matched_control.png",
    "paper/figures/emvr_comparison.png", "paper/figures/risk_coverage.png",
    "paper/figures/qualitative_violation.png",
]
for relative in required_files:
    require((ROOT / relative).exists(), f"required file missing: {relative}")

tex_files = list((ROOT / "paper").rglob("*.tex"))
tex = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
bib = (ROOT / "paper/references.bib").read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex):
    cited.update(key.strip() for key in group.split(","))
require(not (cited - bib_keys), f"undefined citation keys: {sorted(cited - bib_keys)}")
require(not (bib_keys - cited), f"unused bibliography keys: {sorted(bib_keys - cited)}")

labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
require(not (refs - labels), f"undefined LaTeX labels: {sorted(refs - labels)}")

combined_claim_text = (ROOT / "README.md").read_text(encoding="utf-8") + "\n" + tex
for forbidden in (
    "our method improves selective prediction over native confidence",
    "we are the first to introduce ordinal calibration",
    "all examples were manually human-audited",
    "pure background region",
    "Qwen2.5-VL-3B-Instruct-AWQ",
):
    require(forbidden.lower() not in combined_claim_text.lower(), f"unsafe claim/term found: {forbidden}")

markdown_files = [ROOT / "README.md", *list((ROOT / "docs").glob("*.md"))]
for markdown in markdown_files:
    text = markdown.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        if re.match(r"(?:https?://|#|mailto:)", target):
            continue
        require((markdown.parent / target).resolve().exists(), f"broken local link in {markdown.name}: {target}")

if failures:
    print("PAPER VALIDATION FAILED")
    for failure in failures:
        print("-", failure)
    raise SystemExit(1)

print("PAPER VALIDATION PASSED")
print("Accepted manifest SHA-256:", digest)
print("Frozen trajectories/conditions: 176/880")
print("P18 control rows: 850")
print("P19 blur rows: 880")
print("Citation keys:", len(cited))
print("Required files:", len(required_files))
