# Flownomaly V4

This is an ANNAlyzer adaptation of Flownomaly’s conditional flow matching, single-step recovery, and normal-only Jacobian gain-to-noise calibration. It uses DINOv2 ViT-S/14 at the benchmark’s native tile size. Each lighting model starts from the same audited pretrained weights and fine-tunes its own last two ViT blocks and output normalization with masked-token distillation. The fine-tuned encoder is frozen while the flow is fitted. This training protocol differs from the authors’ frozen ViT-G/14 setup. [Paper](https://arxiv.org/abs/2610.32964) · [Upstream implementation](https://github.com/b-rbmp/Flownomaly)

## Run locally

Double-click the launcher in the project’s batch folder to open the lighting checklist. All three lights start selected. Type `1`, `2`, or `3` and press Enter to toggle red, green, or regular. `A` toggles all, an empty Enter runs the selection, and `Q` cancels. For a scripted run, pass one or more lights after `--lighting`. [Run_Flownomaly_V4.bat](../../bat%20files/Run_Flownomaly_V4.bat)

```powershell
& '.\bat files\Run_Flownomaly_V4.bat' check-env
& '.\bat files\Run_Flownomaly_V4.bat' check
& '.\bat files\Run_Flownomaly_V4.bat' smoke
& '.\bat files\Run_Flownomaly_V4.bat' validate --lighting red
```

`smoke` checks the pretrained ViT, trainable-block gradients and flow scoring on synthetic CPU input. The launcher expects its own locked `.venv`, CUDA 12.8 for full runs, and the read-only pretrained DINOv2 weight file already cached by AnomalyDINO. Use `uv sync --locked --project Anamoly_detection/flownomaly --python 3.12.13` to recreate that environment if needed.

## Data and artifacts

Each light’s existing 70 clean training views are partitioned deterministically into 60 fitting views, five flow-checkpoint selection views and five operating-time calibration views. Validation includes both clean and defective frames under the existing benchmark split. Regular keeps the separate varied-normal and fixed-pose defect cohorts. Each completed light has its own fine-tuned `vit_finetuned.pt`, separate `flow_state.pt` and `training.json` under `results/V4/<run>/<lighting>/`. The flow checkpoint stores the SHA-256 of its associated encoder.

Results use the existing 512 px tiles, 128 px overlap, 3 px reflection padding per side, Gaussian stitching, and whole-frame metrics. Run the local report generator after the run completes to see the V4 tab. [Report generator](../../scripts/generate_benchmark_html_report.py)
