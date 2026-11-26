# How the ANNAlyzer models work and how they performed

ANNAlyzer compares models that detect and localize visible blisters on a synthetic injection-moulded battery tray. PatchCore, AnomalyDINO, and U-Net have saved validation results across red, green, and regular illumination. Flownomaly V4 has one completed regular-light validation run. All reported comparisons use the local Blender renders. Performance on real trays remains unmeasured.

## Shared input and scoring

The benchmark takes a native **2048 × 2048 RGB image**. Each model sees **25 overlapping 512 × 512 tiles** with a 384 px stride and 128 px overlap. Tiles are normalized with the backbone's ImageNet channel mean and standard deviation. Each tile yields a continuous 512 × 512 anomaly or probability map. Gaussian stitching with σ = 128 px gives the center of a tile more weight, sums contributions at the original image coordinates, and divides by the sum of the weights at each pixel. Pixel metrics use those continuous scores. The image score is computed separately from the tile scores. [Tiling and Gaussian stitcher](patchcore/V2/phase3.py) · [V3 workflow](v3_workflow.py)

Red and green share grouped 70/15/15 setup IDs stratified by camera/lighting setup. Each setup pairs its clean and defective render within the same fold. Red and green use the same capture setup for a given ID and independent defect seeds. Regular-light defects use the fixed V1 pose. V2 adds 99 new clean regular-light views. The resulting folds have 70 clean views for fitting the regular banks, 15 for threshold calibration, and 15 reserved for test. The three legacy development defects are the regular localization cohort in V1–V3. [Split builder](patchcore/V1/split_dataset.py) · [Frozen split](patchcore/V1/split_v1.json) · [V2 data checks](v2_workflow.py) · [Dataset notes](README.md)

## PatchCore

**What it learns.** A bank of representative feature vectors from clean training tiles. A frozen ImageNet-pretrained WideResNet50-2 extracts layer 2 and layer 3 features. Anomalib pools and aligns the two feature maps and concatenates their channels. The resulting 64 × 64 grid has 4,096 descriptors of 1,536 values each. The selected fit draws two distinct descriptors per spatial stratum from an 8 × 8 partition, yielding 128 candidates per tile. A k-center greedy coreset keeps 10%. With 70 frames this is 224,000 candidates and a **22,400 × 1,536 FP32 bank (131.25 MiB)** per lighting condition. The V1 regular benchmark used one clean image and stored a 320-row bank. The initial regular smoke test used a larger candidate pool than the benchmark. [Backbone and chunked search](patchcore/V2/smoke.py) · [Candidate extraction and coreset](patchcore/V2/phase3.py) · [V1 lighting runner](patchcore/V1/lighting_benchmark.py)

**How it scores.** For a query embedding *q*, each patch score is the Euclidean distance to its nearest memory-bank vector, `min_b ||q − b||₂`. The implementation keeps exact nearest-neighbor search in blocks of 512 queries and 8,192 bank rows. Anomalib reweights the highest-scoring patch using up to nine bank neighbors for the tile score. The frame score is the maximum tile score. For localization, the patch-distance grid is upsampled and smoothed before stitching. Fitting leaves the backbone frozen. This makes PatchCore usable with clean-only data, though bank construction and exhaustive search add memory and time. [V2 model adapter](patchcore/V2/run_v2.py) · [Original smoke implementation](patchcore/V1/smoke.py)

## AnomalyDINO

**What it learns.** A bank of clean patch embeddings from a frozen pretrained DINOv2 ViT-S/14. Each 512 × 512 tile gets 3 px of reflection padding on all four sides. The 518 × 518 input yields a 37 × 37 grid of 1,369 final-layer patch tokens, each with 384 dimensions. Fitting takes 128 spatially stratified tokens per tile and keeps a 10% coreset. With 70 normal frames the bank is **22,400 × 384 FP32 (32.81 MiB)**. In the one-reference V1 regular fit it held 320 tokens. Foreground masking and reference augmentation are disabled. [V1 implementation](anomalydino/V1/run.py) · [V2 encoder and fitting implementation](anomalydino/V2/baseline.py) · [V2 model adapter](anomalydino/V2/run_v2.py)

**How it scores.** Normalize the token vectors and score each with `1 − max_b cosine(q, b)` over its fitted bank. The exact search runs in blocks of 512 queries and 8,192 bank entries. The tile score is the mean of the highest 1% of token distances, and the frame score is the maximum tile score. The upsampled map has its 3 px border removed before stitching. AnomalyDINO uses one quarter of the PatchCore bank storage at the same row count and ran faster in the saved V3 benchmark. Its coarser native token grid is a relevant consideration for very small defects. [Cosine scoring and map generation](anomalydino/V2/baseline.py)

## U-Net

**What it learns.** A supervised ResNet18 encoder–decoder with skip connections and a single logit per output pixel. The encoder starts from ImageNet weights and is trained together with the decoder. The encoder reduces a 512 × 512 tile to a 16 × 16 bottleneck. The decoder upsamples through five stages and recovers spatial details using the skips. The red and green models each use 70 clean and 70 defective source frames. Clean frames use zero masks. Training uses binary cross-entropy plus soft Dice loss, AdamW at `1e-4`, ten epochs, batches of four tiles, CUDA FP16 autocast with gradient scaling, and seed 24117. The highest within-light validation pixel AP selects the checkpoint. Regular uses the selected **red checkpoint as cross-lighting transfer**. It has no regular-light supervised fit. [Architecture and loss](unet/V1/model.py) · [Trainer and checkpoint selection](unet/V1/run.py) · [Frozen V2 regular adapter](unet/V2/run_v2.py) · [V3 transfer workflow](v3_workflow.py)

**How it scores.** Sigmoid converts the logits to pixel probabilities in FP32. Gaussian stitching blends their spatial maps. The tile score is the mean of the highest 1% of tile probabilities, and the maximum over the 25 tiles is the frame score. The model needs labeled defect masks for training and has no nearest-neighbor bank. Its inference cost is independent of the size of the training set once the weights are fitted. [Inference and stitching](unet/V1/run.py)

## Flownomaly V4 adaptation

**What it learns.** A fine-tuned DINOv2 ViT-S/14 encoder plus a time-conditioned neural velocity field. Each lighting condition is independent and uses clean data only. Its 70 outer training normals are split into 60 fit frames, five flow-checkpoint selection frames, and five operating-time calibration frames. During masked-token distillation the last two student transformer blocks and output normalization are updated for three epochs. A frozen teacher supplies targets from unmasked input while the student sees zeroed pixels over roughly 10% of patches. Five tiles per fit frame are sampled per epoch. The encoder is then frozen. Tokens from zero-indexed layers 7, 9, and 11 are concatenated into 1,152-dimensional descriptors on the 37 × 37 grid. Flow fitting samples 64 tokens per tile, standardizes them using the fit subset, and trains the velocity MLP for 12 epochs on interpolations between Gaussian noise and normal tokens. [Encoder, distillation, and velocity field](flownomaly/model.py) · [V4 fitting and cohort code](flownomaly/run_v4.py)

**How it scores.** For a standardized token *q* and sampled noise *z*, the code forms `m = t·q + (1 − t)·z`, applies one recovery step `m + (1 − t)·v(m,t)`, and scores the squared residual to *q*. Operating time *t* is selected from the five normal calibration views using a residual Jacobian gain-to-noise criterion. Tile scores average the highest 1% of the unsmoothed pixel energies after upsampling and cropping. The map path smooths the patch energies, upsamples, removes padding, and uses the shared Gaussian stitcher. The frame score is the maximum tile score. Fitted encoder and flow states have separate filenames and a hash link. The V4 notes record a CPU smoke test for gradient updates, calibration, and exact score/map reload. The completed regular-light V4 run recorded 0.6534 pixel AUROC, 0.0147 pixel AP, 0.3448 AUPRO, and 0.762 s per frame. At the saved image threshold it flagged 0 of 3 defective validation frames. Red and green results remain unavailable. [Configuration and run status](flownomaly/README.md) · [Research journal](JOURNAL.md)

## How to read the performance numbers

- **Pixel AUROC** measures defect-versus-background pixel ranking. **Pixel AP** summarizes the precision–recall curve and is especially informative when defect pixels are rare. V3 red/green pool pixels from both clean and defective frames. Regular pixel metrics pool the three fixed-pose defective frames.
- **AUPRO** integrates mean region overlap through 30% background pixel false-positive rate and normalizes by that limit. Regions are 8-connected mask components. Size bands are small (<64 px), medium (64–511 px), and large (≥512 px). Mean component overlap at 1% FPR is also reported. Both describe localization across components. Individual blister detection recall has no saved result.
- **Image AUROC and AP** rank complete frames. Regular reports show N/A because their clean calibration images and defective images come from different capture protocols. Comparing the two regular cohorts at the saved operating threshold is still useful as a small, explicit cross-cohort check.
- **Image decisions** flag scores strictly above the maximum score among the 15 clean validation frames. The same frames set the threshold and yield zero validation false positives by construction. Held-out false-positive rate and sensitivity remain unmeasured. The raw thresholds use model-specific units.
- **Times** are taken from the saved per-frame inference timers, including stitching. The V3 resource table uses reported inference-phase peak process RAM and CUDA allocated memory. Host RSS measurement methods differ slightly by runner. App startup, file browsing, and overlay comparison are separate from the benchmark latency. [Pixel and region metrics](patchcore/V2/phase3_metrics.py) · [Regular evaluation and threshold](v2_workflow.py) · [Colored evaluation](v3_workflow.py)

## V3 validation results

V3 ran a fresh clean-only bank fit for every lighting condition and retrained the two supervised U-Net models. Red and green each have 15 clean and 15 defective validation frames. Regular has 15 varied clean calibration views and three fixed-pose defective frames. U-Net regular uses the selected V3 red checkpoint.

| Lighting | Model | Pixel AUROC | Pixel AP | AUPRO | Image AUROC | Image AP | Report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Red | PatchCore | 0.9486 | 0.2960 | 0.8177 | 0.9200 | 0.9314 | [JSON](patchcore/results/V3/all_lights_v3_20261124T193302_112321Z/red/validation/report.json) |
| Red | AnomalyDINO | 0.8378 | 0.1346 | 0.6304 | 0.8711 | 0.8857 | [JSON](anomalydino/results/V3/all_lights_v3_20261124T194518_585551Z/red/validation/report.json) |
| Red | U-Net | 0.9090 | 0.7456 | 0.5897 | 0.9733 | 0.9810 | [JSON](unet/results/V3/all_lights_v3_20261124T195634_269837Z/red/validation/report.json) |
| Green | PatchCore | 0.9455 | 0.2930 | 0.8346 | 0.9156 | 0.9220 | [JSON](patchcore/results/V3/all_lights_v3_20261124T193302_112321Z/green/validation/report.json) |
| Green | AnomalyDINO | 0.8317 | 0.1656 | 0.6867 | 0.8667 | 0.8873 | [JSON](anomalydino/results/V3/all_lights_v3_20261124T194518_585551Z/green/validation/report.json) |
| Green | U-Net | 0.9056 | 0.7408 | 0.8099 | 1.0000 | 1.0000 | [JSON](unet/results/V3/all_lights_v3_20261124T195634_269837Z/green/validation/report.json) |
| Regular | PatchCore | 0.9610 | 0.2710 | 0.9071 | N/A | N/A | [JSON](patchcore/results/V3/all_lights_v3_20261124T193302_112321Z/regular/validation/report.json) |
| Regular | AnomalyDINO | 0.8194 | 0.1946 | 0.7163 | N/A | N/A | [JSON](anomalydino/results/V3/all_lights_v3_20261124T194518_585551Z/regular/validation/report.json) |
| Regular | U-Net (red transfer) | 0.9586 | 0.7834 | 0.8198 | N/A | N/A | [JSON](unet/results/V3/all_lights_v3_20261124T195634_269837Z/regular/validation/report.json) |

PatchCore leads pixel AUROC and AUPRO under each lighting condition. U-Net leads pixel AP under each lighting condition and image AUROC and AP on both colored cohorts. Green U-Net separated all 15 clean and 15 defective validation frames, and its pixel AUROC and AP were 0.9056 and 0.7408. These rankings describe different aspects of performance and use one synthetic split.

### Decisions at the saved image thresholds

| Lighting | Model | Threshold | TN | FP | TP | FN | Defective frames flagged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Red | PatchCore | 22.537170 | 15 | 0 | 9 | 6 | 9/15 |
| Red | AnomalyDINO | 0.276754 | 15 | 0 | 8 | 7 | 8/15 |
| Red | U-Net | 0.993900 | 15 | 0 | 14 | 1 | 14/15 |
| Green | PatchCore | 22.310236 | 15 | 0 | 8 | 7 | 8/15 |
| Green | AnomalyDINO | 0.276698 | 15 | 0 | 8 | 7 | 8/15 |
| Green | U-Net | 0.995426 | 15 | 0 | 15 | 0 | 15/15 |
| Regular | PatchCore | 21.543180 | 15 | 0 | 3 | 0 | 3/3 |
| Regular | AnomalyDINO | 0.276206 | 15 | 0 | 1 | 2 | 1/3 |
| Regular | U-Net (red transfer) | 0.999720 | 15 | 0 | 2 | 1 | 2/3 |

### Localization by defect size

Each pair is **AUPRO / mean component overlap at 1% FPR**. The cohorts contain 78 / 58 / 86 small / medium / large components for red, 88 / 49 / 86 for green, and 14 / 15 / 21 for regular.

| Lighting | Model | Small | Medium | Large |
| --- | --- | --- | --- | --- |
| Red | PatchCore | 0.7331 / 0.3256 | 0.8941 / 0.7539 | 0.8430 / 0.5190 |
| Red | AnomalyDINO | 0.5483 / 0.4094 | 0.7903 / 0.6918 | 0.5971 / 0.3259 |
| Red | U-Net | 0.2494 / 0.1965 | 0.6936 / 0.6795 | 0.8282 / 0.8220 |
| Green | PatchCore | 0.7795 / 0.3721 | 0.9038 / 0.7178 | 0.8517 / 0.5536 |
| Green | AnomalyDINO | 0.6865 / 0.5263 | 0.7912 / 0.6532 | 0.6273 / 0.3442 |
| Green | U-Net | 0.7906 / 0.7538 | 0.8056 / 0.7886 | 0.8320 / 0.8174 |
| Regular | PatchCore | 0.9805 / 0.8402 | 0.8959 / 0.8344 | 0.8661 / 0.3890 |
| Regular | AnomalyDINO | 0.8737 / 0.6651 | 0.8069 / 0.5308 | 0.5466 / 0.2726 |
| Regular | U-Net (red transfer) | 0.7121 / 0.6444 | 0.7401 / 0.6383 | 0.9486 / 0.9317 |

Small-defect behavior depends on lighting and operating point. At 1% pixel FPR, AnomalyDINO leads small-region overlap on red, U-Net leads on green, and PatchCore leads on regular. U-Net gives the highest large-region overlap at this operating point across all three lights. The regular cohort contains only three defective frames.

### Compute and storage

Saved V3 runs used an RTX 4060 Laptop GPU with PyTorch 2.9.1+cu128 and CUDA 12.8. RAM is peak inference-process RSS. CUDA is peak allocated memory during the inference measurement. Bank sizes are FP32 embedding storage only.

| Lighting | Model | Mean s/frame | RAM MiB | CUDA MiB | Bank rows × features / MiB |
| --- | --- | --- | --- | --- | --- |
| Red | PatchCore | 1.872 | 2167.6 | 607.4 | 22,400 × 1,536 / 131.25 |
| Red | AnomalyDINO | 0.677 | 1708.8 | 157.5 | 22,400 × 384 / 32.81 |
| Red | U-Net | 0.112 | 1691.5 | 313.8 | N/A |
| Green | PatchCore | 1.868 | 2160.2 | 967.2 | 22,400 × 1,536 / 131.25 |
| Green | AnomalyDINO | 0.647 | 1710.0 | 157.5 | 22,400 × 384 / 32.81 |
| Green | U-Net | 0.113 | 1694.4 | 454.7 | N/A |
| Regular | PatchCore | 1.855 | 2151.8 | 508.1 | 22,400 × 1,536 / 131.25 |
| Regular | AnomalyDINO | 0.647 | 1709.0 | 157.5 | 22,400 × 384 / 32.81 |
| Regular | U-Net (red transfer) | 0.125 | 1686.2 | 454.7 | N/A |

| Lighting | Model | Normal feature extraction (s) | Coreset selection (s) |
| --- | --- | --- | --- |
| Red | PatchCore | 33.66 | 84.06 |
| Red | AnomalyDINO | 47.86 | 85.61 |
| Green | PatchCore | 35.02 | 84.41 |
| Green | AnomalyDINO | 45.39 | 83.50 |
| Regular | PatchCore | 34.33 | 83.78 |
| Regular | AnomalyDINO | 42.80 | 83.18 |

| U-Net lighting | Epoch 1 loss | Epoch 10 loss | Selected epoch | Selected pixel AP | Training passes (s) | Report |
| --- | --- | --- | --- | --- | --- | --- |
| Red | 1.3572 | 0.9113 | 7 | 0.745649 | 1012.06 | [History](unet/results/V3/all_lights_v3_20261124T195634_269837Z/training/red/report.json) |
| Green | 1.3587 | 0.9303 | 9 | 0.740780 | 1009.32 | [History](unet/results/V3/all_lights_v3_20261124T195634_269837Z/training/green/report.json) |

U-Net inference took roughly 0.11–0.13 s per frame in the saved V3 runs, compared with about 0.65–0.68 s for AnomalyDINO and 1.85–1.88 s for PatchCore. The U-Net timings use batched tile inference and mixed precision, while the bank models use FP32 and one tile at a time. [U-Net V3 training curves](unet/results/V3/all_lights_v3_20261124T195634_269837Z/training_curves.png)

## Earlier validation and what changed

### V1 across three lighting conditions

V1 used 70 clean fitting views per colored bank and one fixed regular reference per bank. U-Net trained on 70 clean and 70 defective frames per colored model. Regular validation had three defects and zero independent clean views, so its ranking metrics are pixel-only. Red and green each used 15 clean and 15 defective validation frames.

| Lighting | Model | Pixel AUROC | Pixel AP | AUPRO | Image AUROC | Image AP |
| --- | --- | --- | --- | --- | --- | --- |
| Red | PatchCore | 0.9486 | 0.2960 | 0.8177 | 0.9200 | 0.9314 |
| Red | AnomalyDINO | 0.8378 | 0.1346 | 0.6304 | 0.8711 | 0.8857 |
| Red | U-Net | 0.9090 | 0.7456 | 0.5897 | 0.9733 | 0.9810 |
| Green | PatchCore | 0.9455 | 0.2930 | 0.8346 | 0.9156 | 0.9220 |
| Green | AnomalyDINO | 0.8317 | 0.1656 | 0.6867 | 0.8667 | 0.8873 |
| Green | U-Net | 0.9056 | 0.7408 | 0.8099 | 1.0000 | 1.0000 |
| Regular | PatchCore | 0.8061 | 0.0534 | 0.7015 | N/A | N/A |
| Regular | AnomalyDINO | 0.7437 | 0.0824 | 0.5385 | N/A | N/A |
| Regular | U-Net (red transfer) | 0.9586 | 0.7834 | 0.8198 | N/A | N/A |

[V1 three-model comparison CSV](unet/results/V1/unet_20261124T095027_917613Z/three_model_comparison.csv)

### V1 PatchCore ablation on red

| Candidates/tile | Stitching | Pixel AUROC | Pixel AP | AUPRO | Mean seam step | Mean s/frame |
| --- | --- | --- | --- | --- | --- | --- |
| 64 | Uniform | 0.8638 | 0.1719 | 0.6148 | 0.8460 | 1.178 |
| 64 | Gaussian | 0.8609 | 0.1763 | 0.6041 | 0.3207 | 1.178 |
| 128 | Uniform | 0.9469 | 0.2874 | 0.8141 | 0.7672 | 1.885 |
| 128 | Gaussian | 0.9486 | 0.2960 | 0.8177 | 0.2942 | 1.885 |

With 128 candidates, Gaussian stitching lowered the measured mean boundary step from 0.7672 to 0.2942, about a 62% decrease from uniform stitching. It also produced the highest overall pixel AUROC, AP, and AUPRO in this ablation. The 128-candidate bank doubled from 11,200 to 22,400 rows compared with the 64-candidate bank. Mean inference increased from about 1.18 to 1.89 s/frame. [Ablation report](patchcore/results/V1/red_phase3_20261124T075750_651308Z/report.json)

### V2 regular validation

V2 expanded training of the two regular banks from one to 70 normal views. U-Net used the same red checkpoint as V1. The pixel cohort stayed at the same three fixed-pose defects, so the V1 to V2 regular pixel changes can be compared directly. AUROC/AP at image level are N/A. V2 results available locally are regular only.

| Model | Pixel AUROC | Pixel AP | AUPRO | Δ AUROC (pp) | Δ AP (pp) | Δ AUPRO (pp) | Defective frames flagged | Report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PatchCore | 0.9610 | 0.2710 | 0.9071 | +15.50 | +21.77 | +20.56 | 3/3 | [JSON](patchcore/results/V2/regular_v2_20261124T140600_238105Z/validation/report.json) |
| AnomalyDINO | 0.8194 | 0.1946 | 0.7163 | +7.57 | +11.21 | +17.78 | 1/3 | [JSON](anomalydino/results/V2/regular_v2_20261124T141424_324010Z/validation/report.json) |
| U-Net (red transfer) | 0.9586 | 0.7834 | 0.8198 | 0.00 | 0.00 | 0.00 | 2/3 | [JSON](unet/results/V2/regular_v2_20261124T143148_520300Z/validation/report.json) |

The two banks gained on all three pixel metrics after the regular refit. Both available training-view diversity and bank size increased, so the saved experiment cannot separate their contributions. The threshold adds an operating point using varied clean views while the localization evidence stays on fixed-pose defects.

## Evidence limits and navigation

- **Held-out test performance is unavailable.** The V2 runner defines a test command that reuses the saved validation threshold. I found no saved held-out test report for any model. V3 and V4 result code writes validation results. [Versioned workflow](v2_workflow.py) · [V3 runner](v3_workflow.py) · [V4 runner](flownomaly/run_v4.py)
- **Lighting comparisons depend on capture geometry.** The red/green cohorts share setup IDs but differ in defect instances. Regular defect views use a fixed pose and the clean calibration views vary. The V2 regular protocol accepts a reviewed near-duplicate finding. [Dataset and split audit](JOURNAL.md)
- **Live inspection uses the saved V3 states.** The desktop worker accepts user images, fits other resolutions to the benchmark canvas, and maps continuous scores back to original coordinates. An uploaded ground-truth mask adds exact single-image pixel AP/AUROC using scikit-learn. The saved benchmark metrics use binned pooled scores. The app keeps the classification from the saved image threshold. Scores on resized or real inputs should be interpreted as exploratory. [Inspection worker](inspection_app/worker.py) · [App instructions](inspection_app/README.md)
- **Reports and charts.** The interactive V3 report has model and lighting selectors and matched-frame heatmaps. The workbook includes benchmark tables and charts. [V3 HTML](reports/V3_validation_report.html) · [V3 workbook](reports/excel/V3_results.xlsx) · [Run journal](JOURNAL.md)
