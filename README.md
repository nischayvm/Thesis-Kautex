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
| [Model performance guide](Anamoly_detection/MODEL_PERFORMANCE.md) | How the four models work and what the validation reports show |
| [Tray inspection app](Anamoly_detection/inspection_app/README.md) | Local inference, lighting selection, linked zoom and optional mask comparison |
| [Windows workstation launcher](Anamoly_detection/WORKSTATION.md) | Setup, editable render and training jobs, validation, and reporting |
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

## V3 three-lighting validation

V3 fits PatchCore and AnomalyDINO separately for red, green, and regular illumination using clean training images. U-Net trains separate supervised red and green models and transfers its selected red checkpoint to regular lighting. Red and green each evaluate 15 clean and 15 defective frames from the frozen split. Regular uses 15 varied clean views for threshold calibration and three fixed-pose defects for pixel metrics. Image-ranking metrics for regular lighting are unavailable because the normal and defective cohorts use different capture protocols.

| Lighting | Model | Pixel AUROC | Pixel AP | AUPRO | Image AUROC | Image AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Red | PatchCore | 0.9486 | 0.2960 | 0.8177 | 0.9200 | 0.9314 |
| Red | AnomalyDINO | 0.8378 | 0.1346 | 0.6304 | 0.8711 | 0.8857 |
| Red | U-Net | 0.9090 | 0.7456 | 0.5897 | 0.9733 | 0.9810 |
| Green | PatchCore | 0.9455 | 0.2930 | 0.8346 | 0.9156 | 0.9220 |
| Green | AnomalyDINO | 0.8317 | 0.1656 | 0.6867 | 0.8667 | 0.8873 |
| Green | U-Net | 0.9056 | 0.7408 | 0.8099 | 1.0000 | 1.0000 |
| Regular | PatchCore | 0.9610 | 0.2710 | 0.9071 | N/A | N/A |
| Regular | AnomalyDINO | 0.8194 | 0.1946 | 0.7163 | N/A | N/A |
| Regular | U-Net transfer | 0.9586 | 0.7834 | 0.8198 | N/A | N/A |

AUPRO uses a 30% background-pixel FPR limit. PatchCore led pixel AUROC and AUPRO on all three validation cohorts. U-Net led pixel AP across all three and image AUROC and AP on red and green. These are synthetic validation results.

| Model | Red | Green | Regular |
| --- | --- | --- | --- |
| PatchCore | [Report](Anamoly_detection/patchcore/results/V3/all_lights_v3_20261124T193302_112321Z/red/validation/report.json) | [Report](Anamoly_detection/patchcore/results/V3/all_lights_v3_20261124T193302_112321Z/green/validation/report.json) | [Report](Anamoly_detection/patchcore/results/V3/all_lights_v3_20261124T193302_112321Z/regular/validation/report.json) |
| AnomalyDINO | [Report](Anamoly_detection/anomalydino/results/V3/all_lights_v3_20261124T194518_585551Z/red/validation/report.json) | [Report](Anamoly_detection/anomalydino/results/V3/all_lights_v3_20261124T194518_585551Z/green/validation/report.json) | [Report](Anamoly_detection/anomalydino/results/V3/all_lights_v3_20261124T194518_585551Z/regular/validation/report.json) |
| U-Net | [Report](Anamoly_detection/unet/results/V3/all_lights_v3_20261124T195634_269837Z/red/validation/report.json) | [Report](Anamoly_detection/unet/results/V3/all_lights_v3_20261124T195634_269837Z/green/validation/report.json) | [Report](Anamoly_detection/unet/results/V3/all_lights_v3_20261124T195634_269837Z/regular/validation/report.json) |

Download the HTML report and open it in a browser for interactive metric charts and same-frame heatmaps. The Excel workbook includes editable charts, per-image scores, and training and resource data. Source images, ground-truth masks, and raw score maps remain local. [V3 HTML report](Anamoly_detection/reports/V3_validation_report.html) · [V3 Excel workbook](Anamoly_detection/reports/excel/V3_results.xlsx) · [U-Net V3 training curves](Anamoly_detection/unet/results/V3/all_lights_v3_20261124T195634_269837Z/training_curves.png) · [V3 artifact bundles](Anamoly_detection/benchmark_exports)

## V4 Flownomaly regular validation

The first Flownomaly V4 run used regular lighting. It achieved pixel AUROC **0.6534**, pixel AP **0.0147**, and AUPRO **0.3448**. It flagged **0 of 3** defective validation frames at its saved threshold. Regular image AUROC and AP are unavailable because the clean and defective frames use different capture setups. Download the standalone HTML to see the scores and heatmaps. [V4 HTML report](Anamoly_detection/reports/V4_validation_report.html) · [V4 run report](Anamoly_detection/flownomaly/results/V4/flownomaly_v4_20261125T075129_092983Z/regular/validation/report.json)
