# Thesis-Kautex

This thesis project uses **ANNAlyzer**, a Blender simulation of an industrial inspection cell, to study blister detection on a synthetic automotive battery tray. It generates paired clean and defective renders with pixel-aligned masks for blisters on the mounting flange and cross-ply laminate floor.

## Start here

| Resource | What it covers |
| --- | --- |
| [Dataset and rendering guide](Anamoly_detection/README.md) | Inspection setup, defect generation, lighting conditions and dataset layout |
| [Blender scenes](synthetic_dataset/) and [render scripts](scripts/) | Tray models, inspection-cell controls and batch rendering |
| [PatchCore setup](Anamoly_detection/patchcore/V1/README.md) | Pinned CUDA environment and single-reference smoke test |
| [AnomalyDINO setup](Anamoly_detection/anomalydino/V1/README.md) | Isolated environment, frozen DINOv2 encoder and validation runner |
| [U-Net setup](Anamoly_detection/unet/V1/README.md) | Supervised ResNet18 U-Net training and red-to-regular transfer evaluation |
| [Research journal](Anamoly_detection/JOURNAL.md) | Dataset audit, experiment decisions and measured results |
| [Smoke-test report](Anamoly_detection/patchcore/results/V1/regular_smoke_20261124T072401_611224Z/report.json) | Tensor shapes, timings, memory use and raw scores |
| [Grouped split](Anamoly_detection/patchcore/V1/split_v1.json) and [red validation report](Anamoly_detection/patchcore/results/V1/red_phase3_20261124T075750_651308Z/report.json) | Reproducible setup assignments, coreset comparison and stitching results |

## Dataset

The audited batch contains 300 matched clean image, defective image and binary mask triples at 2048 × 2048. Regular, red and green lighting each have 100 triples. Regular lighting reuses one clean reference. Red and green share the same capture setup for each numeric sample ID and use independent defect seeds.

The PatchCore scripts expect the audited batch at `data/annalyzer_batch_2048/`. The dataset is stored separately from Git. Git LFS tracks the pretrained WideResNet50-2 weights, the regular smoke-test checkpoint and both red validation memory banks.

## PatchCore status

The Anomalib smoke test ran on an RTX 4060 using the single regular-lighting reference. It processed native **512 × 512** tiles with **128 px overlap** and stitched continuous anomaly maps back to **2048 × 2048** for three defective images. The fitted bank had **10,240 × 1,536** FP32 values, and the reloaded checkpoint reproduced the verified map exactly.

Phase 3 froze a 70/15/15 split for red and green using the same setup IDs in each fold. On red validation, **128 spatially stratified candidates per tile**, a **10% coreset**, and **Gaussian stitching** produced pixel AUROC **0.9486**, pixel AP **0.2960**, and AUPRO **0.8177** at a 30% background FPR limit. Gaussian weighting reduced tile-boundary score jumps by about **62%** compared with uniform stitching at the same budget.

The red validation outputs include a side-by-side comparison, full-resolution heatmaps and ground-truth overlays, and the continuous FP32 score maps. The two ZIP archives are tracked with Git LFS. [Visual comparison](Anamoly_detection/patchcore/results/V1/red_phase3_20261124T075750_651308Z/comparison.png) · [Validation figures ZIP](Anamoly_detection/patchcore/results/V1/red_phase3_20261124T075750_651308Z/validation_figures.zip) · [Raw score maps ZIP](Anamoly_detection/patchcore/results/V1/red_phase3_20261124T075750_651308Z/raw_score_maps.zip)

## Validation benchmark

PatchCore, AnomalyDINO and U-Net completed validation on regular, red and green lighting using the frozen 70/15/15 colored setup-ID assignment. PatchCore and AnomalyDINO use normal-only fitting. U-Net uses both normal and labeled defective training images. On regular lighting, PatchCore and AnomalyDINO fit from one clean reference, while U-Net transfers its red-trained checkpoint to the three regular development defects. Regular image-level metrics are N/A because that cohort has zero clean validation frames. The final test sets remain reserved.

| Validation lighting | PatchCore AUPRO | AnomalyDINO AUPRO | U-Net AUPRO | U-Net pixel AP |
| --- | ---: | ---: | ---: | ---: |
| Regular (U-Net transfer) | 0.7015 | 0.5385 | 0.8198 | 0.7834 |
| Red | 0.8177 | 0.6304 | 0.5897 | 0.7456 |
| Green | 0.8346 | 0.6867 | 0.8099 | 0.7408 |

AUPRO is integrated to 30% background false-positive rate. The U-Net checkpoints were selected by same-light validation pixel AP at red epoch 7 and green epoch 9. [Three-model CSV](Anamoly_detection/unet/results/V1/unet_20261124T095027_917613Z/three_model_comparison.csv) · [Metric charts](Anamoly_detection/unet/results/V1/unet_20261124T095027_917613Z/three_model_comparison.png) · [Defect-size comparison](Anamoly_detection/unet/results/V1/unet_20261124T095027_917613Z/three_model_defect_sizes.png) · [Training curves](Anamoly_detection/unet/results/V1/unet_20261124T095027_917613Z/training_curves.png) · [U-Net report](Anamoly_detection/unet/results/V1/unet_20261124T095027_917613Z/report.json)

The two-model results, including image- and pixel-level metrics, are available alongside the charts. [PatchCore report](Anamoly_detection/patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/report.json) · [AnomalyDINO report](Anamoly_detection/anomalydino/results/V1/anomalydino_20261124T091730_300718Z/report.json) · [Comparison chart](Anamoly_detection/anomalydino/results/V1/anomalydino_20261124T091730_300718Z/model_comparison.png) · [Comparison CSV](Anamoly_detection/anomalydino/results/V1/anomalydino_20261124T091730_300718Z/model_comparison.csv)

Git LFS stores the six fitted states listed in the journal and the validation artifact archives. The consolidated PatchCore archive covers the regular and green score maps and the three-color figures. The Phase 3 archive contains the red maps. [PatchCore validation artifacts](Anamoly_detection/benchmark_exports/patchcore_validation_artifacts.zip) · [AnomalyDINO validation artifacts](Anamoly_detection/benchmark_exports/anomalydino_validation_artifacts.zip)

Git LFS also stores the two selected U-Net checkpoints. The full-resolution U-Net maps, masks and overlays are kept with the local run. [Red U-Net checkpoint](Anamoly_detection/unet/results/V1/unet_20261124T095027_917613Z/red/best_validation.pt) · [Green U-Net checkpoint](Anamoly_detection/unet/results/V1/unet_20261124T095027_917613Z/green/best_validation.pt) · [U-Net validation overview](Anamoly_detection/unet/results/V1/unet_20261124T095027_917613Z/lighting_predictions.png)

## V2 regular-light workflow

V2 expands the regular-light normal cohort to 70 training, 15 validation and 15 test views. PatchCore and AnomalyDINO use these training normals for separate fits. U-Net uses the frozen red-to-regular transfer checkpoint. Each model stores V1 and V2 outputs in its own versioned results folders. [Launchers and cohort protocol](Anamoly_detection/README.md#regular-light-v2-models)
