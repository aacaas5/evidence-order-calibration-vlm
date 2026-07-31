# Results Index

Labels:

- **PRIMARY** — paper-facing frozen comparison or consolidated value.
- **CONTROL** — causal sanity/control experiment.
- **ABLATION** — feature/objective diagnostic, not a primary replacement.
- **CACHE** — expensive model output or frozen feature array.
- **INTERMEDIATE** — fold details, progress, QC, or analysis support.

## Main masking phenomenon and features

| Class | File | Contents |
|---|---|---|
| PRIMARY | `results/scaled/p5/summary.json` | severity accuracy/confidence/entropy, native EMVR, violation rate |
| CACHE | `results/scaled/p5/results.json` | 880 frozen Qwen masking rows |
| CACHE | `results/scaled/p6/hidden_features.npz` | 880 × 2050 final frozen features |
| CACHE | `results/scaled/p6/hidden_features_meta.json` | feature row metadata |
| INTERMEDIATE | `results/scaled/p6/fold_results.csv` | grouped BCE folds |
| INTERMEDIATE | `results/scaled/p6/summary.csv` | BCE probe summary |

## Primary masking heads and selective prediction

| Class | File | Contents |
|---|---|---|
| PRIMARY | `results/scaled/p7/summary.csv` | BCE vs BCE+order primary metrics |
| PRIMARY | `results/scaled/p7/emvr_differences.csv` | primary order differences |
| INTERMEDIATE | `results/scaled/p7/fold_results.csv` | fold-wise method results |
| PRIMARY | `results/scaled/p8/risk_coverage_summary.csv` | masking AURC and risks at fixed coverage |
| PRIMARY | `results/scaled/p8/oof_predictions.csv` | out-of-fold native/BCE/order scores |
| INTERMEDIATE | `results/scaled/statistics/bootstrap_summary.json` | masking paired trajectory bootstrap |

## P18 matched non-critical control

| Class | File | Contents |
|---|---|---|
| CONTROL | `results/scaled/p18/control_boxes.json` | deterministic critical/control geometry and validity |
| CONTROL | `results/scaled/p18/control_box_summary.json` | 170 valid, 6 invalid, overlap diagnostics |
| CACHE | `results/scaled/p18/irrelevant_results.json` | 850 matched non-critical Qwen rows |
| CONTROL | `results/scaled/p18/control_metrics.json` | critical/non-critical curves and paired metrics |
| CONTROL | `results/scaled/p18/bootstrap_control.json` | paired 1,000-resample trajectory bootstrap |
| CONTROL | `results/scaled/p18/paired_effect_summary.csv` | per-question paired effects |
| INTERMEDIATE | `results/scaled/p18/control_box_examples.png` | deterministic visual QC grid |

## P19 blur and mask-to-blur transfer

| Class | File | Contents |
|---|---|---|
| CACHE | `results/scaled/p19/blur_results.json` | 880 frozen Qwen blur rows |
| PRIMARY | `results/scaled/p19/blur_summary.json` | blur severity/global phenomenon summary |
| CACHE | `results/scaled/p19/blur_hidden_features.npz` | final blur hidden features |
| CACHE | `results/scaled/p19/blur_hidden_meta.json` | blur feature metadata; category labels may be stale |
| PRIMARY | `results/scaled/p19/transfer_summary.csv` | primary mask-train/blur-test means |
| PRIMARY | `results/scaled/p19/transfer_oof_predictions.csv` | transfer out-of-fold scores |
| INTERMEDIATE | `results/scaled/p19/transfer_fold_results.csv` | fold-wise transfer metrics |
| PRIMARY | `results/scaled/p19/bootstrap_transfer.json` | paired transfer bootstrap intervals |
| INTERMEDIATE | `results/scaled/p19/bootstrap_transfer_samples.csv` | bootstrap samples |
| PRIMARY | `results/scaled/p19/blur_risk_coverage.csv` | blur transfer AURC/risk summary |
| INTERMEDIATE | `results/scaled/p19/blur_qc_examples.png` | blur visual QC |

## P20 ablations and paper consolidation

| Class | File | Contents |
|---|---|---|
| ABLATION | `results/scaled/p20/p20a_ablation_summary.csv` | masking feature/objective ablations |
| ABLATION | `results/scaled/p20/p20a_ablation_fold_results.csv` | masking ablation folds |
| ABLATION | `results/scaled/p20/p20b_transfer_ablation_summary.csv` | mask-to-blur feature/objective ablations |
| ABLATION | `results/scaled/p20/p20b_transfer_ablation_fold_results.csv` | transfer ablation folds |
| PRIMARY | `results/scaled/p20/paper_results_handoff.json` | consolidated source of paper claims and values |
| PRIMARY | `results/scaled/p20/paper_primary_masking.csv` | compact primary masking table |
| PRIMARY | `results/scaled/p20/paper_primary_transfer.csv` | compact primary transfer table |
| PRIMARY | `results/scaled/p20/paper_bootstrap_summary.csv` | compact masking/transfer intervals |
| CONTROL | `results/scaled/p20/paper_control_summary.csv` | compact P18 paired effects |
| PRIMARY | `results/scaled/p20/paper_blur_severity.csv` | exact blur severity curve |
| PRIMARY | `results/scaled/p20/paper_blur_global.csv` | exact native blur EMVR/global summary |
| PRIMARY | `results/scaled/p20/paper_blur_risk_coverage.csv` | compact blur selective prediction table |
| ABLATION | `results/scaled/p20/paper_ablation_masking.csv` | paper-ready P20A table |
| ABLATION | `results/scaled/p20/paper_ablation_transfer.csv` | paper-ready P20B table |
| PRIMARY | `results/scaled/p20/paper_claims_map.csv` | supported/unsupported claim map |

## Audit and taxonomy

| Class | File | Contents |
|---|---|---|
| PRIMARY | `data/gqa/manifests/gqa_evidence_scaled_accepted.json` | frozen 176-trajectory benchmark |
| INTERMEDIATE | `results/p9c/audit_summary.json` | 250-candidate audit summary |
| INTERMEDIATE | `results/scaled/p17/cleanup_report.json` | category/correctness validation report |
| INTERMEDIATE | `results/scaled/p17/category_changes.json` | 205 corrected category labels |
| INTERMEDIATE | `results/scaled/p17/correctness_changes.json` | confirms zero correctness changes |
| INTERMEDIATE | `results/scaled/p17/human_spotcheck_10.json` | limited human spot-check record |

## Source-of-truth rule

Use P20 consolidated files for paper-facing values. If an exploratory artifact differs, investigate rather than choosing the favorable number. Category-specific work must use P17-corrected labels; overall P19 results are unaffected by stale category metadata.
