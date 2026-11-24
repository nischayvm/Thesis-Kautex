# Thesis-Kautex

This thesis project uses **ANNAlyzer**, a Blender simulation of an industrial inspection cell, to study blister detection on a synthetic automotive battery tray. It generates paired clean and defective renders with pixel-aligned masks for blisters on the mounting flange and cross-ply laminate floor.

## Start here

| Resource | What it covers |
| --- | --- |
| [Dataset and rendering guide](Anamoly_detection/README.md) | Inspection setup, defect generation, lighting conditions and dataset layout |
| [Blender scenes](synthetic_dataset/) and [render scripts](scripts/) | Tray models, inspection-cell controls and batch rendering |
| [PatchCore setup](Anamoly_detection/patchcore/README.md) | Pinned CUDA environment and single-reference smoke test |
| [Research journal](Anamoly_detection/JOURNAL.md) | Dataset audit, experiment decisions and measured results |
| [Smoke-test report](Anamoly_detection/patchcore/runs/regular_smoke_20261124T072401_611224Z/report.json) | Tensor shapes, timings, memory use and raw scores |
| [Grouped split](Anamoly_detection/patchcore/split_v1.json) and [red validation report](Anamoly_detection/patchcore/runs/red_phase3_20261124T075750_651308Z/report.json) | Reproducible setup assignments, coreset comparison and stitching results |

## Dataset

The audited batch contains 300 matched clean image, defective image and binary mask triples at 2048 × 2048. Regular, red and green lighting each have 100 triples. Regular lighting reuses one clean reference. Red and green share the same capture setup for each numeric sample ID and use independent defect seeds.

The PatchCore scripts expect the audited batch at `data/annalyzer_batch_2048/`. The dataset is stored separately from Git. Git LFS tracks the pretrained WideResNet50-2 weights, the regular smoke-test checkpoint and both red validation memory banks.

## PatchCore status

The Anomalib smoke test ran on an RTX 4060 using the single regular-lighting reference. It processed native **512 × 512** tiles with **128 px overlap** and stitched continuous anomaly maps back to **2048 × 2048** for three defective images. The fitted bank had **10,240 × 1,536** FP32 values, and the reloaded checkpoint reproduced the verified map exactly.

Phase 3 froze a 70/15/15 split for red and green using the same setup IDs in each fold. On red validation, **128 spatially stratified candidates per tile**, a **10% coreset**, and **Gaussian stitching** produced pixel AUROC **0.9486**, pixel AP **0.2960**, and AUPRO **0.8177** at a 30% background FPR limit. Gaussian weighting reduced tile-boundary score jumps by about **62%** compared with uniform stitching at the same budget.

The red validation outputs include a side-by-side comparison, full-resolution heatmaps and ground-truth overlays, and the continuous FP32 score maps. The two ZIP archives are tracked with Git LFS. [Visual comparison](Anamoly_detection/patchcore/runs/red_phase3_20261124T075750_651308Z/comparison.png) · [Validation figures ZIP](Anamoly_detection/patchcore/runs/red_phase3_20261124T075750_651308Z/validation_figures.zip) · [Raw score maps ZIP](Anamoly_detection/patchcore/runs/red_phase3_20261124T075750_651308Z/raw_score_maps.zip)

Green validation and final test evaluation remain pending. The results so far describe the synthetic inspection cell.
