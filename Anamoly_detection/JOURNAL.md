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

[Setup and run instructions](patchcore/README.md) · [Pinned lockfile](patchcore/uv.lock) · [Machine-readable run report](patchcore/runs/regular_smoke_20261124T072401_611224Z/report.json) · [Heatmaps and ground-truth comparison](patchcore/runs/regular_smoke_20261124T072401_611224Z/comparison.png) · [Fitted state](patchcore/runs/regular_smoke_20261124T072401_611224Z/fitted_state.pt)

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
