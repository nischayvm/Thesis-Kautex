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

## Dataset

The audited batch contains 300 matched clean image, defective image and binary mask triples at 2048 × 2048. Regular, red and green lighting each have 100 triples. Regular lighting reuses one clean reference. Red and green share the same capture setup for each numeric sample ID and use independent defect seeds.

The smoke test expects the audited batch at `data/annalyzer_batch_2048/`. The dataset is stored separately from Git. The pretrained WideResNet50-2 weights and fitted smoke-test checkpoint are tracked through Git LFS.

## PatchCore status

The Anomalib smoke test ran on an RTX 4060 using the single regular-lighting reference. It processed native **512 × 512** tiles with **128 px overlap** and stitched continuous anomaly maps back to **2048 × 2048** for three defective images. The fitted bank had **10,240 × 1,536** FP32 values, and the reloaded checkpoint reproduced the verified map exactly.

The next experiment will evaluate a bounded coreset for red and green lighting. The planned split uses 70 training, 15 validation and 15 test normal images per color, with the same sample IDs assigned to each fold before tiling. This study measures performance in the synthetic inspection cell.
