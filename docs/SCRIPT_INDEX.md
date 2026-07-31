# Script Index

Scripts are intentionally kept in place to avoid breaking relative paths. They fall into four groups: exploratory diagnostics, benchmark preparation, primary experiments, and final paper utilities.

## Final paper utilities

| Script | Role | Cost |
|---|---|---|
| `scripts/p20c_consolidate_results.py` | Consolidate frozen primary, control, transfer, and ablation values | Low |
| `scripts/paper/generate_paper_figures.py` | Regenerate paper figures from cached JSON/CSV only | Low |
| `scripts/paper/validate_paper_results.py` | Check frozen values, paths, citations, and benchmark hash | Low |

## Scaled primary pipeline

| Script | Stage | Role | Cost |
|---|---|---|---|
| `p9a_build_scaled_manifest.py` | P9A | Build 250-candidate scaled manifest | Medium; historical only |
| `p9b_download_scaled_images.py` | P9B | Fetch scaled GQA images | Network/storage |
| `p9c_make_audit_sheets.py` | P9C | Render automated audit sheets | Medium |
| `p9c_finalize_visual_audit.py` | P9C | Finalize accepted/rejected/unsure audit | Low; do not overwrite frozen manifest |
| `p10_p11_run_scaled_qwen.py` | P10–P11 | Frozen Qwen masking inference and hidden extraction | **High** |
| `p12_p14_scaled_models.py` | P12–P14 | Scaled grouped BCE/order heads | Medium |
| `p15_p16_scaled_analysis.py` | P15–P16 | Bootstrap and qualitative failure analysis | Low/medium |
| `p17a_cleanup_scaled_results.py` | P17 | Correct category taxonomy and validate correctness | Low |
| `p17b_select_spotcheck.py` | P17 | Deterministic limited spot-check selection | Low |
| `p18a_build_control_boxes.py` | P18A | Deterministic matched non-critical boxes and QC | Low |
| `p18b_run_irrelevant_qwen.py` | P18B | Frozen Qwen matched-control inference | **High** |
| `p18c_analyze_control.py` | P18C | Paired control metrics/bootstrap/figures | Low |
| `p19a_build_blur_trajectories.py` | P19A | Build local-blur trajectory metadata/QC | Low |
| `p19b_run_blur_inference.py` | P19B | Frozen Qwen blur inference | **High** |
| `p19c_extract_blur_hidden.py` | P19C | Extract blur hidden states | **High** |
| `p19d_mask_train_blur_test.py` | P19D | Grouped mask-train/blur-test transfer | Medium |
| `p20a_final_ablation.py` | P20A | Masking feature/objective ablations | Medium |
| `p20b_transfer_ablation.py` | P20B | Transfer feature/objective ablations | Medium |

## Pilot and exploratory scripts

| Scripts | Purpose |
|---|---|
| `p3_first_evidence_trajectory.py` | Single-example proof of concept |
| `p4c_inspect_real_gqa.py`, `p4d_build_gqa_evidence_manifest.py`, `p4e_selective_fetch_gqa_images.py`, `p4f_manual_gqa_audit.py` | GQA inspection, pilot mapping, image retrieval, and optional audit UI |
| `p5a_real_gqa_trajectory.py`, `p5b_run_gqa_pilot.py`, `p5c_analyze_phenomenon.py` | Real-data pilot trajectory and phenomenon analysis |
| `p6a_extract_hidden_features.py`, `p6b_train_bce_probe.py` | Pilot hidden features and BCE probes |
| `p7_train_order_probe.py`, `p7b_diagnose_order_probe.py`, `p7c_grouped_crossval.py` | Order-head development and grouped validation |
| `p8_selective_prediction.py` | Pilot/scaled selective prediction analysis |

## Diagnostics and maintenance

| Scripts | Purpose |
|---|---|
| `test_qwen_vl.py`, `test_qwen_signals.py`, `test_qwen_hidden_state.py` | Model, token-signal, and hidden-hook viability checks |
| `test_gqa_image_mirror.py`, `test_gqa_mirror_coverage.py`, `test_remote_gqa_zip.py` | Data-host availability diagnostics |
| `inspect_gqa_resources.py`, `inspect_gqa_zip_contents.py`, `check_p4e_report.py` | Local/archive inspection and report validation |
| `download_gqa_pilot_images.py` | Pilot image download helper |
| `fix_p5b_audit_loader.py` | Historical one-off patch helper; retained for provenance, not a pipeline stage |

## Execution cautions

- Run scripts from the repository root because paths are repository-relative.
- Do not rerun P9 finalization over the frozen accepted manifest for paper regeneration.
- Qwen stages save incrementally and are resumable, but may use CPU/disk offload.
- P20 ablations are not the primary headline comparison.
- Several files predate module-level docstrings; this index documents their role without rewriting working experimental code.
