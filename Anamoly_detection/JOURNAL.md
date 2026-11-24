# Anomaly detection journal

## 2026-11-24 · ANNAlyzer dataset audit

Audited `data/annalyzer_batch_2048` before any Anomalib PatchCore fitting. The working folder contains the Blender scenes, render scripts and batch launcher, the completed dataset, and the anomaly-detection README. I read the root `AGENTS.md` and checked the two batch render scripts against the hashes in `dataset.json`. Both matched. The folder had no Git metadata, so there is no revision identifier for this entry. Anomalib was absent from the local Python environment examined. This entry records dataset findings and proposed experiment settings. The dataset was read in place. Dependency installation, split assignment, tile generation and model training were outside this audit.

[Repository instructions](../AGENTS.md) · [Project README](README.md) · [Dataset metadata](../data/annalyzer_batch_2048/dataset.json) · [Source sample metadata](../data/annalyzer_batch_2048/samples.json) · [Audit manifest v1](dataset_audit_manifest_v1.jsonl)

### Inventory, format and pairing

| Lighting | Normal files | Unique normal RGB | Defective RGB | Masks | Complete triples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular | 100 | 1 | 100 | 100 | 100 |
| Red | 100 | 100 | 100 | 100 | 100 |
| Green | 100 | 100 | 100 | 100 | 100 |
| **Total** | **300** | **201** | **300** | **300** | **300** |

- All 900 expected PNGs were present, with zero unexpected data files and zero missing or extra sample records. The root contains the two JSON files and an empty `.batch.lock` marker. `dataset.json` records the batch as complete.
- All PNGs passed Pillow integrity verification and full decoding. Every image is 2048 × 2048. All 600 color images are 8 bits per channel, three-channel RGB PNGs. All 300 masks are 8-bit single-channel grayscale PNGs. Every mask contains foreground and has only values 0 and 255. All foreground counts agree with `samples.json`.
- Each lighting condition has IDs `0001`–`0100`, present in `normal`, `bubbles` and `masks`. All 300 metadata records point to their corresponding files with the same lighting and ID. In every triple, the mean absolute clean-to-defective RGB change was larger inside the mask than outside. As a spot check, overlays at the minimum, middle and maximum mask coverage per lighting condition showed plausible alignment on flange and laminate regions, including around ribs and fastener holes. The nine inspected samples were `regular/{0004,0074,0097}`, `red/{0023,0090,0012}` and `green/{0050,0088,0009}`.
- File-byte SHA-256 and decoded-pixel SHA-256 both found one duplicate group, comprising the 100 regular clean files. This is the reused fixed reference described by the renderer. The batch therefore contains **501 unique RGB images and 300 unique masks**.
- Red and green share all 100 setup IDs. For each same-ID pair, the height band, nominal camera and LED heights, effective LED height, rig and ceiling intensities, and dark-field mix match. Each color has 34 low, 33 mid and 33 high setups, including nine, eight and eight dark-field samples by band. The three conditions share the Front Left camera and tray design. All 300 defective samples have distinct seeds and distinct recorded instance lists. Regular uses one fixed setup, so its numeric ID is a file index and does not identify the same capture setup as red or green.

### Visible defect footprints at source resolution

Measured area as the number of foreground pixels and width and height as the dimensions of the tight axis-aligned bounding box of each 8-connected component. The short side is the smaller of the box width and height. Percentiles use NumPy's linear interpolation. Occlusion and ribs can fragment a single blister, and touching foreground can merge blisters. These measurements describe visible mask regions. The batch provides 3D instance metadata and a binary mask with one foreground class, so the audit reports footprints without assigning 2D regions to specific instances or surfaces. The audit retained five one-pixel components in the statistics.

| Lighting | Visible components | Component area P10 / P50 / P90 (px²) | Median width × height (px)¹ | Short side P10 / P50 / P90 (px) | Foreground per image min / median / max (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Regular | 1,617 | 27 / 162 / 6,065 | 19 × 16 | 5 / 14 / 89 | 2,297 / 25,539.5 / 78,663 |
| Red | 1,481 | 23 / 172 / 7,452 | 20 × 15 | 4 / 13 / 96 | 959 / 28,487.5 / 92,545 |
| Green | 1,452 | 22 / 124.5 / 6,246.2 | 18 × 12 | 4 / 11.5 / 90 | 837 / 26,531 / 85,516 |
| **All** | **4,550** | **24 / 150 / 6,569.8** | **19 × 14** | **4 / 13 / 91** | **837 / 26,508 / 92,545** |

¹ Width and height medians are calculated independently. Across the batch, component areas range from 1 to 41,990 px², widths from 1 to 330 px, heights from 1 to 233 px and component counts from 8 to 23 per mask. The lowest-coverage mask occupies about 0.020% of the full frame, and the median occupies 0.632%.

### Leakage risks and planned split policy

| Configuration | Unique normal training | Unique normal validation | Unique normal testing |
| --- | ---: | ---: | ---: |
| Regular fixed reference | 1 | 0 | 0 |
| Red | 70 | 15 | 15 |
| Green | 70 | 15 | 15 |

- Use three separate Anomalib PatchCore models with the same preprocessing and model configuration. For red and green, the proposed 70/15/15 allocation applies **per lighting condition**. Assign shared numeric setup IDs first, with one assignment reused across both colors and all files in each triple. Preserve representation of the height bands and partial dark-field setups. Only normal training images enter the memory bank. Validation and test use normal and defective samples and their available masks. Any threshold calibration uses validation data only.
- In regular, fit from one unique reference and use defective images for fixed-reference evaluation. Copies associated with held-out anomalies must stay outside the validation and test normal cohorts. Image-level AUROC and an independent normal false-positive rate are unavailable for the regular configuration. Any future defect-based threshold calibration needs held-out defects of its own.
- Split whole source groups before tiling or augmentation. Overlapping tiles from one source share pixels. The same tray, worktop, camera orientation and parts of the lighting schedule recur across samples. Among the distinct red normals, the nearest pair in 64 × 64 RGB thumbnail RMSE was `0091`/`0094` at 4.35 intensity levels on the 0–255 scale. The same IDs were nearest in green at 4.34. This coarse similarity screen flags a pair to review when making assignments. It does not establish a near-duplicate threshold. Recheck exact hashes and visual or image similarity across the assigned folds at that stage.
- Red versus green uses matched capture settings and independently generated defects, so compare performance on similar setup distributions. Regular versus colored runs also change placement, illumination intensity, dark-field use and the size of the clean reference set. All results on this batch characterize one synthetic tray and inspection cell. Generalization to new physical parts or cells requires independent data.

### Initial PatchCore input recommendation

**Use 512 × 512 RGB tiles at native resolution with 25% overlap.** The overlap is 128 px and the stride is 384 px. Tile starts in each axis are 0, 384, 768, 1,152 and 1,536, giving a 5 × 5 grid with full frame coverage and no padding. Crop masks at the same original coordinates. Use identical input geometry for fitting and inference, retain all edge tiles, and start with batch size 1 for the later memory check. For full-image evaluation, use the mean of continuous anomaly-map values in overlapping regions as the initial stitching rule and evaluate on the 2048 × 2048 canvas. The geometric coverage check found that 4,511 of 4,550 component bounding boxes fit entirely in at least one tile. The other 39 span multiple tiles, which makes evaluating the stitched canvas particularly important.

The median component short side is 13 px, with a 10th percentile of 4 px. At a full-frame resize to 256 × 256 these spans project to roughly 1.6 px and 0.5 px. At 512 × 512 full-frame they project to 3.25 px and 1 px. The standard Wide ResNet `layer2` and `layer3` feature grids have nominal strides of 8 and 16 input pixels. One `layer2` interval corresponds to 64 source pixels for a 256 × 256 full-frame input, 32 for a 512 × 512 full-frame input, and 8 with native tiles. Respectively, 3,707 (81.5%), 2,957 (65.0%) and 1,551 (34.1%) of the measured short sides fall below one such interval. This scale comparison is a geometric estimate of localization difficulty, and tiny defects can still challenge the tiled setup. [TorchVision ResNet source](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/resnet.html)

Set Anomalib preprocessing explicitly to preserve each tile at 512 × 512 and retain the full tile field of view. The documented Anomalib default resize is 256 × 256, and the original PatchCore example resizes to 256 then crops to 224. For the later baseline, `AGENTS.md` specifies frozen ImageNet-pretrained `wide_resnet50_2`, `layer2` and `layer3`, and coreset ratio `0.1`. Confirm the actual tensors and feature shapes with a future small fit and inference run. Batch size controls per-step inputs, while total tiling volume also affects the accumulated memory bank. Resolution, throughput and model accuracy remain unbenchmarked. [Anomalib PatchCore reference](https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/patchcore.html) · [Original PatchCore implementation](https://github.com/amazon-science/patchcore-inspection)

### Audit record

The versioned JSON Lines manifest contains one row per verified triple. Paths are relative to `data/annalyzer_batch_2048`. Each row contains its sample identity, synthetic scene and tray design, camera and capture setup, defect seed and instances, file labels, mask pairing, both hash types and measured mask pixel and component counts. Decoded-pixel hashes cover the decoded mode string followed by the row-major pixel bytes. No split field is assigned. The file was reloaded and validated as 300 unique records with the expected paths and mask references.

The audit used Python 3.14 with Pillow 12.1.0, NumPy 2.4.0, OpenCV 5.0.0 and SciPy 1.16.3 already present. The dataset metadata SHA-256 values were as follows.

| File | SHA-256 |
| --- | --- |
| `dataset.json` | `252d185b66dcc29d3503f8f0e8527e1ab3059f6c4b84fb7cd3cded1277fbaa72` |
| `samples.json` | `a80f54f02f8832f4ac830f05759a0e7959fdf55b5726eef8bbb04cd2d6e5bf63` |

## 2026-11-24 · Phase 2 PatchCore feasibility

The regular-lighting smoke test passed on the RTX 4060 Laptop GPU. The isolated environment uses Python 3.12.13, PyTorch 2.9.1+cu128, TorchVision 0.24.1+cu128, CUDA runtime 12.8, cuDNN 9.10.2 and Anomalib 2.2.0. The NVIDIA driver reported 621.14. PyTorch executed an FP32 CUDA operation successfully. The GPU reported 8,187.5 MiB total VRAM and 7,060 MiB free after its startup probe. The host reported 15,554 MiB RAM. Stock Anomalib initially failed to import because its bundled video models implicitly require Requests. Adding Requests to the isolated lock resolved the issue, and the final dependency check passed for all 80 packages.

[Setup and run instructions](patchcore/V1/README.md) · [Pinned lockfile](patchcore/V1/uv.lock) · [Machine-readable run report](patchcore/results/V1/regular_smoke_20261124T072401_611224Z/report.json) · [Heatmaps and ground-truth comparison](patchcore/results/V1/regular_smoke_20261124T072401_611224Z/comparison.png) · [Fitted state](patchcore/results/V1/regular_smoke_20261124T072401_611224Z/fitted_state.pt)

### Smoke test

From `Anamoly_detection/patchcore`, the run command was `.\.venv\Scripts\python.exe -u smoke.py`. The fit used `regular/normal/0001.png` as the sole normal reference. The three defective smoke inference cases were `regular/0004`, `regular/0074` and `regular/0097`, selected to span low, middle and high foreground coverage. The reference, defective images and masks matched the audit manifest hashes. These cases are qualitative diagnostics and were assigned no experimental fold. Dataset metadata hashes still match the audit.

- Extracted 25 native RGB tiles per image using 512 × 512 tiles, 128 px overlap and a 384 px stride. The model received FP32 `[1, 3, 512, 512]` tensors with TF32 disabled. Preprocessing used ImageNet channel mean and standard deviation with no resize or center crop.
- Loaded frozen TorchVision WideResNet50-2 with `IMAGENET1K_V2` weights from the official checkpoint. Its SHA-256 was `9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950`. Observed `layer2` shape `[1, 512, 64, 64]`, `layer3` shape `[1, 1024, 32, 32]`, and 4,096 × 1,536 FP32 embeddings per tile.
- The 25 reference tiles produced 102,400 candidates. With seed 24117 and Anomalib's projected greedy k-center at 0.10, the fitted bank was **10,240 × 1,536 FP32**, or **60 MiB**. The exact chunked Euclidean search passed a comparison with the stock dense search for one and nine neighbors. The reloaded fitted state reproduced all map pixels and tile scores exactly on the `regular/0004` frame.
- All three saved 2048 × 2048 FP32 maps were finite. Raw maps average overlapping tile scores per pixel. Visual PNGs share a single 0–25.7813 display scale across all three images. In the spot check, larger laminate lifts produced distinct peaks, and some small flange regions showed localized responses. Background texture and visible rectangular tile seams also produced responses. Treat seam handling as a validation question before interpreting small-defect localization. Diagnostic means below use the original ground-truth masks, with no threshold calibration or ranking metrics.

| Smoke image | Peak stitched map value | Mean inside mask | Mean outside mask | Frame score¹ |
| --- | ---: | ---: | ---: | ---: |
| `regular/0004` | 19.96 | 11.49 | 6.51 | 21.20 |
| `regular/0074` | 21.55 | 13.50 | 6.56 | 22.55 |
| `regular/0097` | 25.78 | 14.04 | 6.62 | 25.75 |

¹ The frame score is the maximum of Anomalib's weighted tile scores. The raw stitched map and the frame score use separate aggregations.

### Measured resources

| Stage | Wall time | Peak CUDA allocated | Peak CUDA reserved | Peak process RAM |
| --- | ---: | ---: | ---: | ---: |
| Extract 25 normal tiles | 0.75 s | 915 MiB | 996 MiB | 1,726 MiB |
| Fit coreset and verify chunked search | 16.74 s | 1,473 MiB | 1,820 MiB | 1,976 MiB |
| Inference and stitch per 2048 × 2048 frame | 1.11–1.12 s | 453 MiB | 1,822 MiB | 2,096–2,167 MiB |
| Save and reload state, verify one frame | 2.27 s | 524 MiB | 1,832 MiB | 2,302 MiB |
| Save visual outputs | 2.39 s | 432 MiB | 1,832 MiB | 2,316 MiB |

These are synchronized stage timings and PyTorch allocator peaks. Reserved VRAM includes the pool retained from earlier stages. Process RAM is the sampled working set at 50 ms intervals. The report preserves all stages individually, including backbone load. The listed stages plus backbone load total about 26.25 s of measured work, excluding Python imports and input decoding outside the timed stages.

### Red and green feasibility

Estimates are **per color**, based on the measured feature shape. Seventy unique normals per color are planned for fitting. The 100-image column describes the entire available normal inventory for sizing only.

| FP32 memory or count | 70 normals per color | 100 normals per color |
| --- | ---: | ---: |
| Tile count | 1,750 | 2,500 |
| Candidate rows | 7,168,000 | 10,240,000 |
| Raw candidates | 41.02 GiB | 58.59 GiB |
| Stock concatenate lower bound¹ | 82.03 GiB | 117.19 GiB |
| Projected matrix alone¹ | 10.39 GiB | 15.18 GiB |
| Final 10% bank | 4.10 GiB | 5.86 GiB |
| Unchunked distance matrix per tile | 10.94 GiB | 15.63 GiB |
| Chunked distance block alone | 16 MiB | 16 MiB |

¹ The stock implementation retains per-batch tensors while forming the contiguous candidate tensor. K-center then creates the projected matrix and uses additional scratch space. These are tensor storage estimates without Python, model, allocator, or OS overhead. The stock fit exceeds both the host RAM and GPU VRAM budgets. Keeping only a final 10% bank would still consume 4.10 GiB of VRAM per color and incur a large search cost even with distance chunking. [Anomalib 2.2 coreset implementation](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/components/sampling/k_center_greedy.py) · [PatchCore storage and search code](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/patchcore/torch_model.py)

**Recommended next feasibility step.** Process one color at a time and stream a seeded spatially stratified sample of **64 candidates per tile** to host memory. Draw one feature vector per cell of an 8 × 8 grid on the feature map so every tile and every planned normal training view contributes. For 70 images, this bounds the input to 112,000 rows and **656.25 MiB** of original embeddings. A projected greedy k-center pass retaining 10% of those candidates would yield 11,200 rows and **65.63 MiB** of final FP32 bank storage. Keep bounded exact search. Apply the same policy and seed to both colors, and use validation to check localization on small components, seam behavior and the benefit of a larger candidate budget before finalizing the sampling method. These are projected resource bounds for the proposed candidate policy. Phase 2 ran only the single-reference smoke fit and left the original dataset and full-scale experiment untouched.

## 2026-11-24 · Phase 3 grouped split and red-lighting optimization

### Split and experiment design

The frozen split uses seed 24117 and assigns 70 training, 15 validation and 15 test setup IDs to each of red and green. The two colors use the exact same assignment. Each validation and test fold has five setups per height band, including one partial dark-field setup in each band. Normal images, defective images and masks stay together within each colored setup. The previously identified near-duplicate pair `0091` and `0094` stays together in training. The script rebuilds the same assignments from the audited manifest and checks decoded-pixel hashes for leakage across colored folds. [Split manifest](patchcore/V1/split_v1.json) · [Split generator](patchcore/V1/split_dataset.py)

The regular cohort keeps `regular/normal/0001.png` as the one fixed normal reference and excludes the other 99 clean copies. The three inspected defects `regular/0004`, `regular/0074` and `regular/0097` are development/validation images. The other 97 regular defective images are reserved for test. Phase 3 evaluated red validation only. The red, green and regular final test images remained untouched. The green and regular assignments are recorded for future runs.

The two banks used only the same 70 verified red normal images. The pretrained WideResNet50-2, `layer2` and `layer3`, batch size 1, native 512 × 512 tiles, 128 px overlap, FP32, and 10% projected greedy k-center coreset were held fixed. On each 64 × 64 feature grid, an 8 × 8 array of spatial strata supplied one seeded vector per stratum for the 64-candidate pool. The 128-candidate pool added a second distinct vector in each stratum and contained every candidate from the smaller pool. For each bank, both stitchers used the same tile predictions. Uniform averaging gives every overlapping tile the same weight. Gaussian stitching uses separable center weights with sigma 128 px per axis and divides by the local sum of weights. No calibration threshold was fitted.

Pixel AUROC and AP pool full-frame pixels across 15 clean and 15 defective red validation images. AUPRO integrates equally weighted 8-connected component overlap up to a pooled background-pixel FPR of 30% and divides by 0.30. Size bands use component area below 64, 64–511 and at least 512 px². The size-specific curves use the same background FPR and also report mean component overlap at a diagnostic 1% FPR. Metrics use 65,536 shared score bins over the full continuous score range, with a half-resolution sensitivity check. The seam diagnostic compares the mean absolute adjacent-pixel score step at internal tile boundaries against parallel controls 16 px away, on the 15 clean validation images. [AUPRO convention](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/05_research_teaching/Batzner_EfficientAD_Accurate_Visual_Anomaly_Detection_at_Millisecond_Level_Latencies_WACV_2024_paper.pdf)

Run from `Anamoly_detection/patchcore` using the existing Phase 2 environment.

```powershell
.\.venv\Scripts\python.exe split_dataset.py
.\.venv\Scripts\python.exe -u phase3.py
```

### Red validation results

| Candidates per tile | Stitching | Pixel AUROC | Pixel AP | AUPRO at 30% FPR | Mean seam step |
| ---: | --- | ---: | ---: | ---: | ---: |
| 64 | Uniform | 0.8638 | 0.1719 | 0.6148 | 0.8460 |
| 64 | Gaussian | 0.8609 | 0.1763 | 0.6041 | 0.3207 |
| 128 | Uniform | 0.9469 | 0.2874 | 0.8141 | 0.7672 |
| **128** | **Gaussian** | **0.9486** | **0.2960** | **0.8177** | **0.2942** |

There were 457,885 positive and 125,371,235 background pixels. Halving the histogram resolution changed the reported pixel metrics by at most 0.000020. Image AUROC/AP from the maximum weighted tile score were 0.8311/0.8554 with 64 candidates and 0.9200/0.9314 with 128. This image score is shared by both stitchers.

| Candidates and stitch | Small AUPRO / overlap¹ (78) | Medium AUPRO / overlap¹ (58) | Large AUPRO / overlap¹ (86) |
| --- | ---: | ---: | ---: |
| 64 uniform | 0.472 / 0.180 | 0.800 / 0.647 | 0.619 / 0.326 |
| 64 Gaussian | 0.454 / 0.176 | 0.798 / 0.645 | 0.609 / 0.328 |
| 128 uniform | 0.731 / 0.341 | 0.889 / 0.751 | 0.839 / 0.514 |
| 128 Gaussian | 0.733 / 0.326 | 0.894 / 0.754 | 0.843 / 0.519 |

¹ Mean component overlap at 1% pooled background FPR. The counts in parentheses refer to visible mask components. The 128-candidate Gaussian setup led on aggregate and size-specific AUPRO. Uniform stitching with 128 candidates retained a 1.5 percentage-point advantage in small-component overlap at the 1% diagnostic FPR. Small components remained the hardest group for either stitcher.

At 128 candidates, Gaussian weighting lowered the mean seam step by 61.7%. Relative to the nearby control step, the seam dropped from 10.46× to 4.17×. The comparison shows softer rectangular boundaries and persistent responses on tray ribs and background structures. The full-resolution heatmaps share one visual scale from 0 to 27.2401. Metrics use the raw stitched scores. Extract the archives into the run directory to recover its `visuals/` and `raw/` outputs. [Visual comparison](patchcore/results/V1/red_phase3_20261124T075750_651308Z/comparison.png) · [Validation figures ZIP](patchcore/results/V1/red_phase3_20261124T075750_651308Z/validation_figures.zip) · [Raw score maps ZIP](patchcore/results/V1/red_phase3_20261124T075750_651308Z/raw_score_maps.zip) · [Run report](patchcore/results/V1/red_phase3_20261124T075750_651308Z/report.json)

### Measured resources

Shared extraction from the 70 red training normals took 34.12 s. The peak sampled process RAM was 3,038 MiB, and the PyTorch allocator peaked at 345 MiB allocated and 398 MiB reserved. Both candidate budgets were extracted in this one pass.

| Candidates per tile | Candidate vectors | Fitted bank | Coreset time | Fit RAM peak | Fit VRAM allocated / reserved | Mean inference and stitch | Inference RAM peak |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | 112,000 | 11,200 × 1,536 · 65.63 MiB | 20.13 s | 3,149 MiB | 2,070 / 2,112 MiB | 1.18 s/frame | 3,532 MiB |
| 128 | 224,000 | 22,400 × 1,536 · 131.25 MiB | 83.35 s | 3,360 MiB | 3,883 / 3,918 MiB | 1.89 s/frame | 3,592 MiB |

Shared extraction plus coreset took 54.25 s for the 64-candidate budget and 117.47 s for 128. Per-frame inference includes tile inference and one stitch and excludes PNG decoding, map writes, and metric calculations. The 128 inference stage peaked at 510 MiB allocated and 632 MiB reserved. The first run completed fitting and inference but stopped at a type error during overlay export. The visualization fix passed an end-to-end synthetic check, then the full comparison completed in a new directory. Both saved banks and spot-checked 64-candidate raw maps matched the first attempt byte for byte.

### Configuration selected on red validation

Use **128 spatially stratified candidates per tile, a 10% coreset, and Gaussian stitching with sigma 128 px per axis**. The larger bank improved localization across size bands and fitted within the RTX 4060 memory budget. Gaussian stitching improved all three pooled pixel metrics at that budget and substantially reduced the measured seam step. Check the selected configuration on the matched green validation fold before drawing cross-lighting conclusions. The red result is based on 15 synthetic validation setups using one seed. Final test evaluation remains reserved.

## 2026-11-24 · Preliminary PatchCore benchmark across lighting conditions

### Fixed configuration and cohorts

Used the selected Phase 3 configuration for both new fits and carried forward the saved red validation scores and bank. The shared pipeline uses frozen ImageNet-pretrained WideResNet50-2 `layer2` and `layer3`, native 512 × 512 RGB tiles with a 128 px overlap, ImageNet channel normalization, FP32 with TF32 disabled, batch size 1, 128 spatially stratified candidates per tile, a 10% projected greedy k-center coreset, nine-neighbor weighted tile scores, exact chunked search, seed 24117, and Gaussian stitching with sigma 128 px per axis. The shared finite-value assertion now checks candidates in chunks to avoid a large temporary GPU allocation. Candidate selection and coreset computation are unchanged. [Frozen split](patchcore/V1/split_v1.json) · [Selected red reference report](patchcore/results/V1/red_phase3_20261124T075750_651308Z/report.json) · [Benchmark runner](patchcore/V1/lighting_benchmark.py)

Red and green each use the same 70 training capture IDs and 15 validation capture IDs, with one clean and one defective image per validation setup. Regular uses only `regular/normal/0001.png` for training. Its three validation defects are `regular/0004`, `regular/0074` and `regular/0097`. Regular image AUROC and image AP are N/A because the regular protocol has no independent clean validation frame. Its pixel metrics use foreground and background pixels from the three defective frames. This differs from the red and green pixel cohorts, which also include 15 clean frames each. The split and source dataset were retained as recorded. The reserved test cohorts remained outside fitting and validation.

All frames use the Front Left inspection camera. Regular has a fixed camera height of 1.610 m, a nominal LED height of 1.102 m, base rig and ceiling intensities of 1.0, and no dark-field mixing. The colored training sets share camera heights of 1.281–1.608 m and nominal LED heights of 1.103–1.592 m. The colored validation sets share camera heights of 1.315–1.605 m and nominal LED heights of 1.118–1.536 m. Effective LED heights in the colored validation sets range from 1.084–1.536 m, and three of the 15 validation setups use partial dark field. The two colored cohorts match in placement and recorded intensity schedule at each ID and generate defects independently. The regular condition additionally differs in training coverage, view geometry and the size of its validation cohort.

The benchmark uses the Phase 3 full-frame metrics, including AUPRO at 30% pooled background FPR and size-specific component overlap at a diagnostic 1% FPR. Each score histogram uses 65,536 bins spanning its validation score range, with the saved red metrics retained at their original bin range. Scores remain continuous in the saved maps. A shared display scale is used for the three-color heatmap comparison. No operating threshold was calibrated.

Run from `Anamoly_detection/patchcore` after the Phase 2 environment is installed. The runner expects the existing red raw maps and overlays alongside the Phase 3 report. The archived copies can be extracted into the original red run directory.

```powershell
.\.venv\Scripts\python.exe -u lighting_benchmark.py
```

### Consolidated validation results

| Lighting | Training normals | Validation normal / defective | Pixel AUROC | Pixel AP | AUPRO at 30% FPR | Image AUROC | Image AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Regular | 1 | 0 / 3 | 0.8061 | 0.0534 | 0.7015 | N/A | N/A |
| Red | 70 | 15 / 15 | 0.9486 | 0.2960 | 0.8177 | 0.9200 | 0.9314 |
| Green | 70 | 15 / 15 | 0.9455 | 0.2930 | 0.8346 | 0.9156 | 0.9220 |

| Lighting | Recorded flange / laminate instances | Visible mask components | Positive pixels |
| --- | ---: | ---: | ---: |
| Regular | 27 / 18 | 50 | 106,761 |
| Red | 125 / 82 | 222 | 457,885 |
| Green | 131 / 74 | 223 | 482,793 |

Instance totals count generated geometry. Connected components count visible mask regions, which can split around ribs or occlusions. Red and green use the same setup schedule with independently seeded defects, so their modest metric differences reflect both lighting color and differences in the rendered defects. Across the matched setup cohorts, red had slightly higher pixel and image ranking scores, and green had higher region overlap. The regular heatmaps showed strong responses along ribs, rims and background edges. Its small cohort, single clean reference and fixed setup limit direct comparison with the colored cohorts.

| Lighting | Small components¹ | Medium components¹ | Large components¹ |
| --- | ---: | ---: | ---: |
| Regular | 14 · 0.843 / 0.150 | 15 · 0.743 / 0.394 | 21 · 0.578 / 0.102 |
| Red | 78 · 0.733 / 0.326 | 58 · 0.894 / 0.754 | 86 · 0.843 / 0.519 |
| Green | 88 · 0.779 / 0.372 | 49 · 0.904 / 0.718 | 86 · 0.852 / 0.554 |

¹ Each cell gives the component count, AUPRO at 30% FPR, and mean overlap at 1% FPR. Size bands use visible area below 64, 64–511 and at least 512 px². Regular's small-component AUPRO is based on 14 regions and its mean small-component overlap at 1% FPR is 0.150. Halving histogram resolution changed the reported pooled pixel metrics by at most 0.000020 across the three results.

### Fitted models and artifacts

| Lighting | Candidate vectors | Fitted bank | Extraction / coreset time | Mean validation inference and stitch |
| --- | ---: | --- | ---: | ---: |
| Regular | 3,200 | 320 × 1,536 · 1.875 MiB | 1.68 / 0.50 s | 0.67 s/frame |
| Red | 224,000 | 22,400 × 1,536 · 131.25 MiB | 34.12 / 83.35 s | 1.89 s/frame |
| Green | 224,000 | 22,400 × 1,536 · 131.25 MiB | 34.92 / 85.22 s | 1.88 s/frame |

The red resource measurements are from Phase 3. Green peaked at 2,206 MiB allocated VRAM during coreset selection and 3,432 MiB sampled process RAM during extraction. The consolidated report stores per-stage allocator and working-set measurements and the hashes of all three standalone fitted states. The red state was packaged with the saved Phase 3 bank and original backbone weights. Its historical metrics, raw scores and bank were retained.

[Comparison CSV](patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/comparison.csv) · [Consolidated report](patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/report.json) · [Validation metric chart](patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/validation_metrics.png) · [Defect-size chart](patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/defect_size_metrics.png) · [Heatmaps and ground-truth panel](patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/lighting_heatmaps.png)

[Fitted regular model](patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/regular/fitted_state.pt) · [Fitted red model](patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/red/fitted_state.pt) · [Fitted green model](patchcore/results/V1/lighting_benchmark_20261124T084630_908213Z/green/fitted_state.pt)

## 2026-11-24 · Model 2 AnomalyDINO

The PatchCore freeze records SHA-256 hashes for its implementation, split, lockfile, environment metadata, pretrained weights, and all files in the three completed runs. AnomalyDINO uses its own `.venv`, package lock, model cache and output directories. The AnomalyDINO runner reads the frozen split, cohort records, stitcher, evaluator and saved PatchCore report. [Freeze manifest](anomalydino/V1/patchcore_freeze_v1.json) · [AnomalyDINO setup](anomalydino/V1/README.md) · [Runner](anomalydino/V1/run.py)

The AnomalyDINO environment uses Python 3.12.13, Anomalib 2.5.1, timm 1.0.27, PyTorch 2.9.1+cu128 and TorchVision 0.24.1+cu128. The backbone is the frozen DINOv2 ViT-S/14 checkpoint `vit_small_patch14_dinov2.lvd142m`, pinned by revision and SHA-256. Fitting uses normal samples only, with foreground masking and reference augmentation disabled. Anomalib 2.5.1's k-center includes its seeded initial center in the returned bank. [Anomalib AnomalyDINO reference](https://anomalib.readthedocs.io/en/lib-v2.5.1/markdown/guides/reference/models/image/anomaly_dino.html) · [Pinned encoder weights](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/blob/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors)

AnomalyDINO keeps the original 512 × 512 tile coordinates, 128 px overlap, ImageNet RGB normalization, FP32, batch size 1, seed 24117 and Gaussian stitching with sigma 128 px. Reflection padding adds three pixels per side so the encoder receives 518 × 518 while preserving every native pixel. The last transformer block supplies 37 × 37 tokens of width 384. The border is removed from each generated map before stitching. For each tile the fit takes 128 spatially stratified tokens, two per cell of an 8 × 8 partition of the token grid, and applies the 10% projected greedy k-center coreset. Cosine search uses one nearest neighbor with 512-query and 8,192-bank chunks. AnomalyDINO's tile score is the mean of the top 1% of token distances. The frame score is the maximum of its tile scores. These model-specific choices replace PatchCore's convolutional features, Euclidean search and weighted tile score.

The CUDA smoke test fitted 12 bank vectors from one red normal tile and evaluated a red validation tile. It confirmed the 518 × 518 encoder input, 1,369 × 384 patch features, and aligned 512 × 512 output. The maximum difference between chunked and stock dense Anomalib maps was 3.58e-7. Reloading the saved state reproduced the smoke map exactly. The full runs use the identical red, green and regular validation cohorts and pixel metric definitions as PatchCore. Regular image AUROC and AP are N/A because its validation fold has zero independent clean frames. The final test sets remain reserved.

### Validation results

| Lighting | Model | Pixel AUROC | Pixel AP | AUPRO at 30% FPR | Image AUROC | Image AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Regular | PatchCore | 0.8061 | 0.0534 | 0.7015 | N/A | N/A |
| Regular | AnomalyDINO | 0.7437 | 0.0824 | 0.5385 | N/A | N/A |
| Red | PatchCore | 0.9486 | 0.2960 | 0.8177 | 0.9200 | 0.9314 |
| Red | AnomalyDINO | 0.8378 | 0.1346 | 0.6304 | 0.8711 | 0.8857 |
| Green | PatchCore | 0.9455 | 0.2930 | 0.8346 | 0.9156 | 0.9220 |
| Green | AnomalyDINO | 0.8317 | 0.1656 | 0.6867 | 0.8667 | 0.8873 |

| Lighting | Small AUPRO / overlap¹ | Medium AUPRO / overlap¹ | Large AUPRO / overlap¹ |
| --- | ---: | ---: | ---: |
| Regular | 0.706 / 0.157 | 0.614 / 0.200 | 0.373 / 0.127 |
| Red | 0.548 / 0.409 | 0.790 / 0.692 | 0.597 / 0.326 |
| Green | 0.687 / 0.526 | 0.791 / 0.653 | 0.627 / 0.344 |

¹ These are AnomalyDINO results. Each cell gives AUPRO at 30% pooled background FPR and mean visible-component overlap at a diagnostic 1% FPR, using the same <64, 64–511, and ≥512 px² bins as PatchCore. Halving score-histogram resolution changed AnomalyDINO's pooled metrics by at most 0.000009.

PatchCore led on aggregate AUPRO in all three conditions and on both colored image-level metrics. AnomalyDINO had higher pixel AP on the three regular defects. At the stricter 1% background FPR, AnomalyDINO also recovered more pixels within small components on red (0.409 versus 0.326 mean overlap) and green (0.526 versus 0.372). These observations describe fixed synthetic validation cohorts and the selected single configuration for each method.

### Fitted states and resources

| Lighting | Candidate vectors | Fitted bank | Candidate extraction / coreset | Mean inference and stitch |
| --- | ---: | ---: | ---: | ---: |
| Regular | 3,200 | 320 × 384 · 0.469 MiB | 0.89 / 0.08 s | 0.59 s/frame |
| Red | 224,000 | 22,400 × 384 · 32.81 MiB | 44.48 / 83.88 s | 0.66 s/frame |
| Green | 224,000 | 22,400 × 384 · 32.81 MiB | 43.99 / 83.90 s | 0.67 s/frame |

The RTX 4060 run peaked at 955 MiB of allocated CUDA memory and 1,993 MiB of sampled process RAM. Red and green inference took about one third of the measured PatchCore time on the same cohorts. The saved state for each condition includes its frozen encoder and fitted bank. Each bank was confirmed after reloading the state. The report includes individual scores, runtime and environment measurements, and model hashes. The display heatmaps use a common AnomalyDINO range. The raw maps preserve the original cosine-distance score scale.

Ran `.\.venv\Scripts\python.exe -u run.py` from `Anamoly_detection/anomalydino`. Rechecking the PatchCore freeze after the benchmark matched all 447 fingerprints.

[Consolidated report](anomalydino/results/V1/anomalydino_20261124T091730_300718Z/report.json) · [Cross-model metrics CSV](anomalydino/results/V1/anomalydino_20261124T091730_300718Z/model_comparison.csv) · [Cross-model metric chart](anomalydino/results/V1/anomalydino_20261124T091730_300718Z/model_comparison.png) · [Defect-size chart](anomalydino/results/V1/anomalydino_20261124T091730_300718Z/defect_size_comparison.png) · [Heatmaps and ground-truth panel](anomalydino/results/V1/anomalydino_20261124T091730_300718Z/lighting_heatmaps.png)

[Fitted regular state](anomalydino/results/V1/anomalydino_20261124T091730_300718Z/regular/fitted_state.pt) · [Fitted red state](anomalydino/results/V1/anomalydino_20261124T091730_300718Z/red/fitted_state.pt) · [Fitted green state](anomalydino/results/V1/anomalydino_20261124T091730_300718Z/green/fitted_state.pt)

## 2026-11-24 · Model 3 supervised U-Net

PatchCore and AnomalyDINO are the frozen comparison references. Their implementation, environment metadata, fitted states and completed outputs are fingerprinted separately. U-Net uses its own pinned Python 3.12.13, PyTorch 2.9.1+cu128 and TorchVision 0.24.1+cu128 environment. [PatchCore freeze](anomalydino/V1/patchcore_freeze_v1.json) · [AnomalyDINO freeze](unet/V1/anomalydino_freeze_v1.json) · [U-Net setup](unet/V1/README.md)

The encoder is ResNet18 initialized with TorchVision ImageNet1K V1 weights. The U-Net decoder uses skip connections and fixed channel widths of 128, 64, 32, 16 and 8 from coarse to fine. Both encoder and decoder are trainable. Loss is BCE with logits plus per-tile soft Dice loss with smoothing 1 and equal coefficients. AdamW uses learning rate 1e-4 and weight decay 0.01. Each lighting fit uses 10 epochs, batch size 4, CUDA FP16 autocast and gradient scaling, FP32 loss and maps, and seed 24117. The batch-size-4 forward and backward smoke test passed at native tile size.

Red and green each use 70 normal and 70 defective audited source images from the frozen training IDs. Each epoch visits all 3,500 native training tiles in a seeded shuffle. Clean targets are all zero, and defective targets use the source binary masks. The 15 clean and 15 defective frames of each colored validation fold are used to select the highest full-frame pixel AP, with the earliest checkpoint winning ties. Regular is a separately labeled red-to-regular transfer evaluation on the three existing development defects. Zero regular frames enter fitting or checkpoint selection. The final test folds remain reserved.

All three methods share the frozen split and validation cohorts, 512 × 512 source coordinates, 128 px overlap, Gaussian stitching with sigma 128 px, and the same pixel and connected-component metric definitions. U-Net's one-channel logits become probabilities before stitching. Image score is the maximum across tiles of the mean of the top 1% of tile probabilities. Pixel AP selection and final U-Net metrics use 65,536 bins across the full [0, 1] probability range. Prior results retain their original score ranges. Regular image AUROC and AP are N/A for all three methods because the validation cohort has zero independent normals. The U-Net transfer also differs in which lighting trained the model and in labelled defect supervision.

Run from `Anamoly_detection/unet` using the locked environment.

```powershell
.\.venv\Scripts\python.exe -u run.py
.\.venv\Scripts\python.exe -u render_results.py <run-directory>
```

### Three-model validation comparison

| Validation lighting | Model and training protocol | Pixel AUROC | Pixel AP | AUPRO at 30% FPR | Image AUROC | Image AP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Regular | PatchCore, fixed normal reference | 0.8061 | 0.0534 | 0.7015 | N/A | N/A |
| Regular | AnomalyDINO, fixed normal reference | 0.7437 | 0.0824 | 0.5385 | N/A | N/A |
| Regular | U-Net, red-to-regular transfer | 0.9586 | 0.7834 | 0.8198 | N/A | N/A |
| Red | PatchCore, normal-only | 0.9486 | 0.2960 | 0.8177 | 0.9200 | 0.9314 |
| Red | AnomalyDINO, normal-only | 0.8378 | 0.1346 | 0.6304 | 0.8711 | 0.8857 |
| Red | U-Net, supervised red | 0.9090 | 0.7456 | 0.5897 | 0.9733 | 0.9810 |
| Green | PatchCore, normal-only | 0.9455 | 0.2930 | 0.8346 | 0.9156 | 0.9220 |
| Green | AnomalyDINO, normal-only | 0.8317 | 0.1656 | 0.6867 | 0.8667 | 0.8873 |
| Green | U-Net, supervised green | 0.9056 | 0.7408 | 0.8099 | 1.0000 | 1.0000 |

U-Net had the highest pixel AP and colored image-level scores on these validation folds. PatchCore had the highest pixel AUROC and aggregate AUPRO on both colored folds. In particular, red U-Net was effective on the largest mask regions and had substantially lower overlap on small regions. The three-frame regular transfer result uses the red model selected on red validation. Its supervised source training and the smaller regular validation cohort make it a separate protocol from the one-reference regular baselines. The perfect green image ranking is based on the 30 synthetic validation frames.

| U-Net validation lighting | Small AUPRO / overlap¹ | Medium AUPRO / overlap¹ | Large AUPRO / overlap¹ |
| --- | ---: | ---: | ---: |
| Regular transfer | 0.712 / 0.644 | 0.740 / 0.638 | 0.949 / 0.932 |
| Red | 0.249 / 0.196 | 0.694 / 0.679 | 0.828 / 0.822 |
| Green | 0.791 / 0.754 | 0.806 / 0.789 | 0.832 / 0.817 |

¹ AUPRO integrates to 30% background FPR. Overlap is the mean component overlap at the diagnostic 1% FPR. Size bands are <64, 64–511 and ≥512 px². Regular, red and green contain 50, 222 and 223 visible validation mask regions, respectively.

### Checkpoint selection and resource use

| Fit | Best epoch | Best validation pixel AP | Training time | Epoch validation time | Peak process RAM | Peak CUDA allocated / reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Red | 7 | 0.745649 | 1,041.37 s | 46.59 s | 4,249 MiB | 656 / 760 MiB |
| Green | 9 | 0.740780 | 1,019.71 s | 42.28 s | 4,233 MiB | 797 / 890 MiB |

Training times sum the 10 measured training passes and exclude data loading, validation and checkpoint writes. Mean inference including stitching was 0.114 s per frame for red, 0.110 s for green and 0.109 s for regular transfer. U-Net used FP16 autocast and up to four tiles per inference batch. The earlier methods used one tile per batch. Both selected U-Net checkpoints reproduced their epoch pixel AP when reloaded. The transfer report records the exact red checkpoint hash and epoch. Final train BCE + Dice decreased from 1.357 to 0.911 for red and from 1.359 to 0.930 for green.

The completed run contains 63 native 2048 × 2048 FP32 probability arrays and the full-resolution ground-truth masks, ground-truth overlays, probability heatmaps and prediction overlays for all 33 defective validation frames. Both frozen baseline inventories matched after U-Net. [Three-model CSV](unet/results/V1/unet_20261124T095027_917613Z/three_model_comparison.csv) · [Validation comparison chart](unet/results/V1/unet_20261124T095027_917613Z/three_model_comparison.png) · [Size-specific comparison](unet/results/V1/unet_20261124T095027_917613Z/three_model_defect_sizes.png) · [Training curves](unet/results/V1/unet_20261124T095027_917613Z/training_curves.png) · [Prediction and ground-truth panel](unet/results/V1/unet_20261124T095027_917613Z/lighting_predictions.png) · [Detailed U-Net report](unet/results/V1/unet_20261124T095027_917613Z/report.json)

[Best red checkpoint](unet/results/V1/unet_20261124T095027_917613Z/red/best_validation.pt) · [Best green checkpoint](unet/results/V1/unet_20261124T095027_917613Z/green/best_validation.pt)

## 2026-11-24 · Regular-light normals v2 handoff

The v2 render adds 99 defect-free views for a normal-only regular-light experiment. The original batch, v1 audit, frozen split, and model checkpoints remain the reference for the published v1 comparisons. The v1 regular cohort has one unique normal image and 99 copies. The v2 manifest references `regular/normal/0001.png` once as the original training reference. It adds 69 new training views, 15 validation views, and 15 test views for a planned total of **70/15/15 clean views**.

The v2 batch was still running at 18:45 IST. The checkpoint contained 10 of 99 new images plus the original reference, and `dataset.json` reported `complete: false`. The final duplicate audit had not run. No v2 models were fitted or evaluated for this entry. [Render status](../data/annalyzer_regular_normals_v2/dataset.json) · [v2 manifest](../data/annalyzer_regular_normals_v2/manifest_v2.json) · [v1 split](patchcore/V1/split_v1.json)

### Data and split contract

- Use `data/annalyzer_regular_normals_v2/manifest_v2.json`. Its object keys are the numeric view IDs and `reference`. Resolve each `path` from the project root. Rows have `image_id`, `group_id`, `split`, `label: normal`, and `mask_path: null`. New views include the evaluated camera matrix, lens, LED height, and image hash. The original reference stays in the v1 directory.
- The v2 images are 2048 × 2048 RGB PNGs from the Front Left camera. Camera height and nominal LED height reuse the related red/green setup IDs. The v2 rig uses the regular RGB color, fixed rig and ceiling intensities, and dark-field mix 0. A matching ID means the samples are correlated for split purposes. Differences in intensity, dark field, and effective LED height prevent a wavelength-only comparison.
- The 99 new views inherit the frozen red/green fold assignments. Training setup `colored/0100` was omitted so the original regular reference could occupy one training slot. Keep inherited group IDs together across any joint experiment. The new views have no same-setup defect render or mask, even when the numeric ID matches a v1 filename.
- Before fitting, verify `complete: true`, the 100 manifest rows, the source hashes, image dimensions, and the final duplicate report. The renderer checks exact decoded RGB duplicates and flags cross-fold thumbnail RMSE at or below 2.5 on a 0–255 scale. Treat that cutoff as a screen. Review flagged pairs and the nearest cross-fold pair before freezing an ML manifest. Record any approved changes in a new split version and keep the v1 split intact. [v2 renderer](../scripts/render_annalyzer_normals_v2.py)

### Proposed next experiment

1. **Regular PatchCore v2.** Fit from the verified v2 training normals only. Reuse the pinned v1 backbone, feature layers, native 512 × 512 tiles with 128 px overlap, 10% coreset, and Gaussian stitching. Run a local fit/inference and reload smoke test before scaling. Save the new bank and reports in a separate run directory. Keep the v1 regular model for a matched comparison and preserve the existing red and green models.
2. **Evaluate by cohort.** Use the v2 validation normals to set a false-alarm operating point and the v2 test normals to check false positives across the new heights. Report localization on the existing fixed-pose regular defects separately, using the reserved splits. Image-level AUROC/AP for the varied setup are unavailable until it has held-out positives from that setup. A matched evaluation would use 15 new defective/mask renders at the v2 validation poses and 15 at the v2 test poses, paired with the existing clean views. Any calibration that uses labeled validation defects must be recorded.
3. **Other models.** Fit AnomalyDINO on the same verified v2 training normals after the PatchCore smoke test. Keep the current U-Net's red-to-regular transfer result as a separate baseline. A regular-light supervised fit would require labeled regular defects in the training fold. The current v2 output is a normal-only supplement. [Frozen v1 PatchCore configuration](patchcore/V1/lighting_benchmark.py) · [v1 U-Net protocol](unet/V1/run.py)

## 2026-11-24 · V2 model launchers and verification

Archived each model's V1 scripts, configuration and local weight files in `V1/`, and moved run folders with their fitted states to `results/V1/`. The workspace-level `V1/` backup stays separate. The archived V1 scripts retain their original path assumptions. Each new `V2/` runner reuses V1 extraction, stitching and evaluation code and writes its own timestamped run to `results/V2/`. Installed environments and caches remain at the model roots. The renderer still reads the byte-identical compatibility copy of `patchcore/split_v1.json`. [Frozen split](patchcore/V1/split_v1.json) · [Launcher instructions](README.md#regular-light-v2-models)

| Launcher | V2 regular-light procedure |
| --- | --- |
| [PatchCore](../Run_PatchCore_V2.bat) | One-tile fit and reload smoke check, then normal-only fitting from the 70 V2 training views. Uses WideResNet50-2 `layer2` and `layer3`, FP32, 128 stratified candidates per tile and a 10% coreset. |
| [AnomalyDINO](../Run_AnomalyDINO_V2.bat) | One-tile fit and reload smoke check, then fitting on the same 70 views. Uses frozen DINOv2 ViT-S/14, FP32, 518 × 518 reflection padding, chunked cosine search, 128 stratified candidates per tile and a 10% coreset. Removes padding before stitching. |
| [U-Net](../Run_UNet_V2.bat) | Loads and verifies the red epoch-7 validation checkpoint. Runs a forward smoke check and red-to-regular transfer with the V1 FP16 autocast inference settings. Regular V2 provides no labeled training defects for a supervised fit. |

Each runner preserves the 2048 × 2048 frame, native 512 px tiles, 128 px overlap, Gaussian stitching with sigma 128 px, and seed 24117. The existing red and green V1 runs supply the colored baselines. All launchers support `check-env`, `check`, and `validate`, with validation as the default. `test "<run-folder>"` verifies the saved validation inputs, source code, calibration and checkpoint before scoring the reserved cohorts. `all` explicitly requests both stages.

Preflight checks the completed render and 70/15/15 assignments against the frozen split, verifies image hashes and PNG structure, and checks the renderer's duplicate audit. Flagged near-duplicate pairs require a recorded review decision. Validation uses the maximum score across the 15 varied-pose validation normals as the fixed image threshold, with scores strictly above it counted as alarms. The validation false-positive rate is therefore zero by construction. The 15 held-out varied normals estimate false positives, in steps of one frame. The three validation and 97 reserved fixed-pose defects provide separate sensitivity and localization results. These defects share the original reference geometry, while the V2 normal views vary in camera and LED height. Image AUROC and AP for the varied-pose cohort are unavailable because it has no matched positive captures.

Runs save the checkpoint, data fingerprints, smoke diagnostics, continuous stitched score maps, per-image scores, representative heatmaps and overlays, resource measurements, and image-score charts. Fixed-pose localization uses pixel AUROC, pixel AP, and AUPRO to 30% background FPR with the V1 connected-component size bands. Test uses the validation display scale for visual comparisons.

**Verified during setup.** All three batch launchers parsed help and rejected the incomplete live manifest before any fitting. Their isolated environments detected the RTX 4060 Laptop GPU with PyTorch 2.9.1+cu128 and verified the required weight files. The 862 V1 model files checked against the backup inventory matched their SHA-256 hashes. The three historical validation RGB/mask pairs also matched their hashes and mask counts. Synthetic checks exercised the complete 70/15/15 preflight, near-duplicate review guard, validation threshold reuse, and result export.

**Render status at this check.** The live manifest held 81 of 100 records, including all 70 training views and 11 validation views. The renderer still reported `complete: false`, so the final duplicate audit and live V2 preflight remain pending. No V2 model fit, dataset inference, threshold calibration, or reserved test evaluation ran. [Render status](../data/annalyzer_regular_normals_v2/dataset.json)

### Completed render and reviewed split

The renderer completed 99 new PNGs plus the V1 reference. Preflight verified 70 training, 15 validation and 15 test normals and the three fixed-pose validation image/mask pairs. The renderer found zero decoded-pixel duplicates and flagged four cross-fold pairs at its 2.5 thumbnail-RMSE threshold.

| V2 setup IDs | Folds | Thumbnail RMSE | Native RGB RMSE |
| --- | --- | ---: | ---: |
| `0079` / `0068` | Train / validation | 2.31 | 6.76 |
| `0082` / `0098` | Train / validation | 2.41 | 8.85 |
| `0096` / `0075` | Train / validation | 2.42 | 8.73 |
| `0094` / `0087` | Train / test | 2.48 | 7.36 |

Each pair had different camera and LED heights. Visual inspection confirmed close views of the same synthetic tray. Retain the frozen assignment for the preliminary benchmark and disclose the train/test similarity when reporting held-out false alarms. The three launchers now use this review decision when double-clicked. Acceptance is bound to the checked manifest and audit hashes. All three passed the live read-only preflight with the reviewed split. No fitting, inference, or test evaluation ran during this change. [Duplicate audit](../data/annalyzer_regular_normals_v2/dataset.json) · [Launcher instructions](README.md#regular-light-v2-models)
