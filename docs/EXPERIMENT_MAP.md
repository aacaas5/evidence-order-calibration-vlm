# Experiment Map

The project evolved through pilots, scaled evaluation, controls, transfer, and final consolidation. “Complete” means a cached artifact exists; it does not imply every stage belongs in the main paper.

| Stage | Scientific question | Main input | Main output | Status | Paper role |
|---|---|---|---|---|---|
| P0 | Can the selected VLM run in the local environment? | Qwen checkpoint, test image | viability checks | Complete / exploratory | None; setup provenance |
| P1 | Can answer-token confidence and entropy be extracted correctly? | single VLM generations | signal diagnostics | Complete / exploratory | Method provenance |
| P2 | Can a stable final-layer representation be captured? | Qwen forward/generation states | hidden-vector diagnostics | Complete / exploratory | Method provenance |
| P3 | Does a synthetic/local example expose a confidence trajectory? | one image, critical box | first five-state trajectory and images | Complete / proof of concept | Motivation/qualitative provenance |
| P4 | Can GQA questions be mapped to defensible critical objects? | GQA questions and scene graphs | pilot manifest, downloads, audit artifacts | Complete | Dataset construction |
| P5 | Does the phenomenon occur on real GQA pilot data? | audited pilot manifest | VLM rows and phenomenon summaries | Complete / pilot | Preliminary only |
| P6 | Can frozen features support correctness prediction? | P5/P10 generations and hidden states | feature matrices, BCE probe results | Complete | Reliability-head foundation |
| P7 | Does the order objective reduce violations? | grouped trajectories and frozen features | BCE/order predictions and metrics | Complete | Primary masking method result |
| P8 | Do learned scores improve risk--coverage? | out-of-fold P7 predictions | AURC and risk--coverage curves | Complete | Main negative result |
| P9 | Can the benchmark be scaled and visually screened? | GQA balanced validation + scene graphs | 250 candidates, audit, frozen 176 accepted | Complete / frozen | Main benchmark provenance |
| P10–P11 | What are the scaled answers, signals, and hidden features? | frozen accepted manifest + images | 880 result rows, 2,050-D features | Complete / cache | Primary data cache |
| P12–P14 | How do scaled BCE/order heads behave? | scaled features | grouped CV predictions and summaries | Complete | Primary method pipeline |
| P15 | Are method differences supported under clustered resampling? | grouped out-of-fold predictions | trajectory bootstrap CIs | Complete | Main statistical evidence |
| P16 | What qualitative failures and successes occur? | cached trajectories and images | representative cases and figure | Complete | Analysis/appendix |
| P17 | Are category and correctness labels internally consistent? | scaled outputs | corrected taxonomy and spot-check | Complete | Taxonomy correction; no correctness change |
| P18 | Is degradation specific to critical evidence? | accepted manifest, scene graphs, cached critical rows | matched non-critical boxes/results/bootstrap | Complete / control | Main causal sanity check |
| P19 | Does mask-trained order supervision transfer to local blur? | mask features + blur trajectories | blur inference, transfer predictions/bootstrap | Complete | Main transfer result |
| P20A | Which masking features/objectives matter? | frozen masking features | ablation tables | Complete / ablation | Compact paper ablation |
| P20B | Which features/objectives transfer to blur? | frozen mask/blur features | transfer ablation tables | Complete / ablation | Objective attribution |
| P20C | What are the frozen paper-facing values and claims? | P7/P8/P18/P19/P20 artifacts | consolidated CSV/JSON handoff | Complete / primary index | Paper source of truth |
| P21 | What is the defensible novelty boundary? | targeted literature review through Aug. 2026 | verified positioning and bibliography | Complete for draft | Related Work / claim safety |

## Frozen boundaries

- The P9 accepted manifest remains unchanged at 176 trajectories.
- P17 taxonomy is used for any category-specific analysis.
- P18 invalid controls are not removed from the main benchmark.
- P20 ablations do not replace primary P7/P19 values.
- P21 positioning does not claim first ordinal VLM calibration or first evidence masking.
