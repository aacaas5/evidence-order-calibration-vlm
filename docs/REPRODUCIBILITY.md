# Reproducibility

This document separates inexpensive paper regeneration from full VLM inference. The numerical claims in the paper come from frozen cached outputs; no new inference is required to compile the manuscript.

## Frozen scientific scope

- Model: `Qwen/Qwen2.5-VL-3B-Instruct` (non-AWQ)
- Answer model: frozen in all main experiments
- Accepted manifest: `data/gqa/manifests/gqa_evidence_scaled_accepted.json`
- Accepted manifest SHA-256: `FD991EF4101D105CBB80F256B9CE0001C83DF2529E02C93E3B0B1CE4C11820EA`
- Accepted trajectories: 176
- Main masking rows: 880 (five severities per trajectory)
- Severities: `0.00, 0.25, 0.50, 0.75, 1.00`
- Fixed order-head settings: `mu=0.3`, `margin=0.05`, `L2=1.0`, `seed=42`

Do not regenerate, edit, filter, or replace the accepted manifest when reproducing paper tables. P18's six invalid controls are excluded only from that paired control.

## Recorded environment

Confirmed in the executed `.venv` on 2026-08-29:

| Component | Version/details |
|---|---|
| OS | Windows 11 / PowerShell environment |
| Python | 3.14.5 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| Transformers | 5.16.1 |
| Accelerate | 1.14.0 |
| qwen-vl-utils | 0.0.14 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| GPU memory | 8,151 MiB |
| Driver | 592.82 |

The Qwen loading scripts use `device_map="auto"`. On this 8 GB-class GPU, model parameters may be offloaded to CPU and disk; this is expected and makes inference slower. The scripts use deterministic generation (`do_sample=False`) and save results incrementally.

PyTorch CUDA wheels may require the platform-specific index documented by PyTorch rather than the default package index. Install the matching CUDA build first if `pip install -r requirements.txt` resolves a CPU-only wheel.

## Data acquisition and licensing

GQA questions/scene graphs and Visual Genome-derived images are upstream research datasets. They are not intended to be committed to this repository. Obtain them under the original dataset terms and place them as follows:

```text
data/gqa/metadata/val_balanced_questions.json
data/gqa/metadata/val_sceneGraphs.json
data/gqa/scaled_images/<image_id>.jpg
```

The local `data/gqa/archives/`, `data/gqa/metadata/`, and image directories are ignored by Git. Manifests remain visible because they define the controlled benchmark. See the GQA and Visual Genome citations in `paper/references.bib`.

## Recommended cached-output workflow

This path regenerates the paper consolidation and figures without running Qwen:

```powershell
.venv\Scripts\python.exe scripts/p20c_consolidate_results.py
.venv\Scripts\python.exe scripts/paper/generate_paper_figures.py
.venv\Scripts\python.exe scripts/paper/validate_paper_results.py
Set-Location paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

With Perl available, `latexmk -pdf -interaction=nonstopmode main.tex` can run
the same dependency-aware build in one command.

Primary inputs are under `results/scaled/p20/`. The consistency validator checks row counts, expected values, paper/README claims, citation keys, required files, and the accepted-manifest hash.

## Full experiment stages

Run from the repository root. Review each script before execution; several stages are expensive and some early stages are historical/pilot work.

### Benchmark construction and audit (historical; do not overwrite the frozen accepted manifest)

```powershell
.venv\Scripts\python.exe scripts/p9a_build_scaled_manifest.py
.venv\Scripts\python.exe scripts/p9b_download_scaled_images.py
.venv\Scripts\python.exe scripts/p9c_make_audit_sheets.py
.venv\Scripts\python.exe scripts/p9c_finalize_visual_audit.py
```

These commands describe provenance. For paper reproduction, use the frozen accepted manifest instead of rerunning selection/audit.

### Main masking inference and frozen features (expensive)

```powershell
.venv\Scripts\python.exe scripts/p10_p11_run_scaled_qwen.py
.venv\Scripts\python.exe scripts/p12_p14_scaled_models.py
.venv\Scripts\python.exe scripts/p15_p16_scaled_analysis.py
```

`p10_p11_run_scaled_qwen.py` writes answer/confidence rows and 2,050-dimensional frozen feature vectors. It excludes all tokenizer special/control tokens from sequence confidence and entropy.

### Category correction and spot-check

```powershell
.venv\Scripts\python.exe scripts/p17a_cleanup_scaled_results.py
.venv\Scripts\python.exe scripts/p17b_select_spotcheck.py
```

P17 changed 205 category labels and zero correctness labels. Use corrected P17 taxonomy for category-specific claims. Some P19 metadata retains stale pre-P17 categories; overall metrics are unaffected.

### Matched non-critical control (Qwen inference is expensive)

```powershell
.venv\Scripts\python.exe scripts/p18a_build_control_boxes.py
.venv\Scripts\python.exe scripts/p18b_run_irrelevant_qwen.py
.venv\Scripts\python.exe scripts/p18c_analyze_control.py
```

P18 has 170 valid and 6 invalid same-size controls. The inference runner safely resumes from `results/scaled/p18/irrelevant_results.json`.

### Held-out blur transfer (Qwen/feature extraction is expensive)

```powershell
.venv\Scripts\python.exe scripts/p19a_build_blur_trajectories.py
.venv\Scripts\python.exe scripts/p19b_run_blur_inference.py
.venv\Scripts\python.exe scripts/p19c_extract_blur_hidden.py
.venv\Scripts\python.exe scripts/p19d_mask_train_blur_test.py
```

Blur is confined to the critical box with radius `min(24, 0.15 * lambda * min(width, height))`. Training uses masking trajectories only; blur question IDs are held out by grouped cross-validation.

### Final ablations and consolidation

```powershell
.venv\Scripts\python.exe scripts/p20a_final_ablation.py
.venv\Scripts\python.exe scripts/p20b_transfer_ablation.py
.venv\Scripts\python.exe scripts/p20c_consolidate_results.py
```

P20A/P20B are ablations. Do not substitute their values for the frozen primary P7/P19 comparisons.

## Determinism and statistics

- Qwen generation: greedy/deterministic.
- Model/head seed: 42.
- Bootstrap seed: 42.
- Bootstrap resamples: 1,000.
- Bootstrap unit: whole paired question trajectory, never an individual severity row.
- Correctness: strict normalized exact answer matching.

Minor floating-point differences may occur across CUDA/library builds. Paper claims should be checked against `results/scaled/p20/paper_results_handoff.json`, not silently replaced by a more favorable rerun.

## Large local artifacts

The following are intentionally local or ignored:

- GQA archives, metadata dumps, and images: large and subject to upstream terms.
- Hugging Face model caches/checkpoints: regenerated from the official model identifier.
- resumable `hidden_progress.npz` and P19 partial hidden-state files: transient checkpoints.

Final compact JSON/CSV results, paper figures, and final compressed feature arrays are retained locally. If distributing the repository without feature arrays, rerunning head ablations requires first regenerating the frozen hidden features with the expensive extraction stages.
