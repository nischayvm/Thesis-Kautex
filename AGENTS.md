# AGENTS.md — Industrial Computer Vision and ML Research

## Scope and working style
- Act as a computer vision research engineer working on surface anomaly detection and localization on injection-molded plastic parts. PatchCore is the initial baseline, not a permanent restriction on model choice.
- Inspect the existing repository, environment, scripts, and data before proposing changes. Reuse working code and avoid unnecessary abstractions or unrelated refactors.
- Work in small, independently verifiable steps. Explain the objective, proposed changes, assumptions, and verification for each step. Report actual results, failures, and limitations; never invent metrics.
- Ask before full training/fitting, long rendering, extensive sweeps, installing or upgrading dependencies, downloading weights, destructive changes, or sending data externally. Short local read-only checks and agreed smoke tests are permitted.
- Keep company data, CAD, imagery, model artifacts and credentials local. No external uploads, telemetry, or remote inference without explicit authorization. Check model, weight, dataset, and dependency licenses before company use.

## Data integrity and split protocol
- Treat source renders, masks, CAD and Blender assets as immutable. Write derived data and results to separate paths; never silently overwrite prior runs.
- Initial dataset is expected under `data/annalyzer_batch_2048/`, with `regular`, `red`, and `green` folders containing `normal`, `bubbles`, and `masks`. Verify actual paths, counts, formats, dimensions, IDs and pairings; do not assume completeness or infer sample counts from the total PNG count.
- Create a versioned manifest with image ID, underlying scene/part/defect identity, lighting, camera pose, label, mask path and available render metadata. Verify what IDs represent using the render metadata.
- Group matched lighting variants, normal/defective counterparts and other correlated views or tiles of the same underlying sample into the same split. Split original groups before tiling or augmentation; check exact and near-duplicates across splits.
- Use only verified normal training images for normal-only anomaly models. Never fit the memory bank, feature normalization or learned preprocessing on validation/test data. Held-out anomalies and masks may be used for validation and evaluation, not for normal-only fitting.
- Keep validation and final test distinct. Select preprocessing, coreset ratio, hyperparameters and operating thresholds on validation only. Record immutable split manifests and seeds. Explicitly disclose any use of labeled anomalies for threshold selection.
- Inspect representative image/mask overlays, empty or missing masks, nonbinary mask values, channel order, resolution, defect sizes and background/part boundaries before fitting. Fail loudly on invalid pairings; do not silently discard samples.

## PatchCore baseline
- Prefer a maintained, version-pinned implementation (e.g. Anomalib) and compare behavior with the original Amazon Science implementation. Inspect installed API and dependencies before writing code.
- Begin with a frozen ImageNet-pretrained `wide_resnet50_2` backbone, `layer2` and `layer3` features, and coreset sampling ratio `0.1`. Use the selected implementation's documented scoring/neighbor defaults; record the exact settings, since implementations differ.
- PatchCore fits a normal patch-feature memory bank and a coreset rather than optimizing the backbone with gradient descent. Do not introduce supervised losses, epochs, optimizers or defect training examples merely to imitate supervised pipelines.
- Record backbone/weight identity, feature layers, input transforms, aggregation, coreset algorithm and size, nearest-neighbor method, image scoring, map interpolation/smoothing and threshold calibration.
- First run a small normal-only fit and held-out inference smoke test. Check that features, memory bank, anomaly maps and image scores are finite and correctly shaped. Save/reload the fitted state and verify inference consistency before scaling.
- Use raw continuous anomaly scores for ranking metrics. Do not independently min-max normalize every image for quantitative evaluation. Separate display-only heatmaps from saved quantitative anomaly maps.
- Reference: https://github.com/amazon-science/patchcore-inspection
- Reference: https://anomalib.readthedocs.io/en/stable/markdown/guides/reference/models/image/patchcore.html

## Image resolution and hardware
- Preserve original 2048 x 2048 images where verified. Measure defect footprints in original pixels before choosing a model input size; verify they remain visible after resizing and at the feature-map stride.
- Use a conventional low-resolution baseline only as a documented reference, not as proof of adequate small-defect detection. Evaluate higher input resolution or native-resolution overlapping tiles when needed.
- For tiled processing, record tile size, stride, padding, overlap and edge handling. Reassemble continuous anomaly maps in original coordinates with a fixed, documented overlap rule; evaluate full images, not independent tiles. Use identical tiling policy in fitting and inference unless explicitly justified.
- Preserve aspect ratio and field of view; do not silently crop relevant regions. Apply geometric transforms consistently to images, masks and camera metadata when applicable. Use nearest-neighbor interpolation for discrete masks.
- Use checkpoint-required RGB channel order and normalization. Do not use color jitter, grayscale conversion, per-image color normalization, or channel mixing by default during illumination comparison.
- Target the available local GPU (currently RTX 4060 with 8 GB VRAM; verify runtime hardware). Start with batch size 1 for high-resolution tests. Track GPU peak memory, host RAM, memory-bank size and runtime. Use bounded/chunked nearest-neighbor search where supported.
- On OOM, report the failing stage and propose a controlled adjustment (batch size, chunk size, coreset or tiling). Never silently change resolution, sampling ratio, precision or model architecture.

## Illumination experiments
- Start with one lighting condition for an end-to-end baseline, then fit separate regular, red and green models using matched group IDs and equal normal-training counts where available.
- Use identical split assignments, preprocessing policy, model configuration and evaluation protocol across conditions. Report per-condition results before any pooled model experiment.
- Treat pooled-lighting fitting and cross-lighting testing as separate experiments with explicit labels. Do not mix colors in the initial single-condition baseline.
- Audit camera pose, LED height, exposure, geometry and defect identity across conditions. If regular uses fixed placement while colored renders vary, acknowledge the confound; only claim isolated wavelength effects from otherwise matched setups.
- Keep RGB data even for monochromatic illumination unless a different channel policy is tested as a controlled experiment.

## Evaluation and reproducibility
- Report image-level AUROC and average precision (AP), pixel-level AUROC and AP, and AUPRO with its false-positive-rate integration limit (e.g. 0.30). Specify definitions, averaging, evaluation resolution and whether normal images are included; report metrics as unavailable if prerequisites are missing.
- At a validation-selected operating threshold, report image precision, recall, F1, false-positive rate and confusion matrix; report mask Dice/IoU where meaningful. Do not optimize thresholds on test data or label test-max F1 as deployed performance.
- Inspect small-defect performance, false alarms on ribs/edges/reflections, and representative failures. Save original image, ground truth, continuous anomaly map, overlay and binary prediction separately. Declare whether evaluation covers the whole frame or a predefined part ROI; never use defect masks to define an inference ROI.
- Record config, split manifest, seed, library versions, Git revision, hardware, runtime, checkpoint/weight identifier, memory bank and per-image scores. Preserve results under uniquely named run directories.
- Establish one functioning baseline before adding new architectures. For comparative claims, control tuning budgets, report seed variation or group-level uncertainty where feasible, and disclose all changed settings.
- Synthetic performance does not establish real-world performance. Validate on independent real parts when available; distinguish visible surface defects from internal voids that an RGB image cannot establish.

## Definition of done
- Data and mask checks pass; splits are grouped and reproducible; no anomalous training inputs or test leakage.
- A small end-to-end fit/inference test passes and actual outputs are inspected before full experiments.
- For each completed experiment, provide the exact run command, configuration, split counts, metrics, output paths, failures and next recommended step.
- For later 2D-to-3D work, preserve original pixel coordinates and camera metadata; state calibration, coordinate frames, scale source and mapping uncertainty. Do not claim 3D localization from a 2D anomaly map alone.
