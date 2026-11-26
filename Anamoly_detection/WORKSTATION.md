# Windows workstation launcher

Double-click [ANNAlyzer_Master.bat](../ANNAlyzer_Master.bat) in the repository root to open the PyQt6 workstation window. If PyQt6 is missing, the bootstrap terminal prompts to install the isolated UI environment first. The launcher starts on Python 3.10 or newer, including Python 3.14. Model setup uses the existing lockfiles with **isolated Python 3.12.13 and the pinned CUDA 12.8 PyTorch builds**. The report and UI environments use separate requirements files. [Pinned report requirements](../scripts/workstation-requirements.txt) · [Pinned UI requirements](inspection_app/requirements.txt)

## First run

On the work PC, clone the private repository while signed in to GitHub. The workstation setup can fetch pretrained weights separately, so you can skip downloading the historical Git LFS bundles:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/nischayvm/Thesis-Kautex.git
Remove-Item Env:\GIT_LFS_SKIP_SMUDGE
Set-Location Thesis-Kautex
.\ANNAlyzer_Master.bat
```

1. Use the **Settings** dropdown to edit paths, rendering, split, training, and model options. Click **Save** to write the default job file at `.workstation/jobs/default.json`. Use **Save as…** for another preset and **Open…** to load it. Paths can be absolute or relative to the repository root.
2. Select **Install requirements** from the **Action** dropdown. Tick the model and reporting environments you need, then click **Run**. This installs the selected components under `.workstation/envs/`. A first install needs access to the package repositories and Python downloads.
3. Choose **Download or verify pretrained weights**. Downloads use the checkpoint URLs and SHA-256 hashes from the existing implementations. If the workstation is offline or uses an internal mirror, place the pinned files in the paths listed in the model setup docs.
4. Choose **Check GPU and environments**. Check the actual GPU name, available memory, driver and CUDA operation before scheduling long jobs. The reports record the actual device used. The launcher runs selected GPU jobs one at a time.

The **Job terminal** streams setup, Blender, model and reporting output in the window. Its status line shows completion, failure or cancellation. Use **Send** to answer prompts such as the publishing selection. Each run's terminal output is also saved under `.workstation/logs/`. Use **Stop job** to stop the active job and its child processes.

## Choosing settings

| Settings | Editable values |
| --- | --- |
| Job and Blender | Dataset folder, scene path, `blender.exe` path, OptiX device name substring. Blank device name selects the first available OptiX GPU. |
| Render | Selected lights, matched clean/defective pairs per light, square resolution, seed, Cycles samples, optional fixed camera and LED heights in metres. A pair includes its binary mask. Regular custom jobs render clean and defective views at the same setup. |
| Dataset split | Training and validation percentages, test remainder, grouped seed, and explicit acceptance flag after reviewing cross-fold perceptual matches. |
| Shared training | Model and light lists, seed, overlap, Gaussian sigma and PyTorch CPU threads. The 512 px tile stays fixed. The last tile moves to the frame edge when necessary. |
| PatchCore and AnomalyDINO | Spatial candidate count per tile, coreset ratio and query/bank chunk sizes. |
| U-Net | Epochs, batch size, learning rate and FP16/BF16/FP32. Custom regular jobs train on pairs from regular lighting. The original V3 benchmark keeps red-to-regular transfer. |
| Flownomaly | Encoder epochs and learning rate, flow epochs and learning rate, flow batch size, and tokens per tile. Flow fitting, checkpoint selection and calibration use separate subsets of the training normals. |

Lighting and model lists use checkboxes. Text fields accept Windows paths with spaces. Direct JSON edits use the same validation and use list syntax such as `["red", "green"]`.

## Running and finding results

Run **Render**, then **Audit**, then **Fit and validate** for a custom job. The audit verifies complete image/mask pairs, binary mask contents, exact RGB duplicates, and cross-fold image similarity candidates. Review any flagged candidates in the job's `.split.json` before accepting them. The split groups identical setup IDs across lights and leaves a held-out test fold.

Fit writes uniquely named model runs under `Anamoly_detection/<model>/results/V5/` with the resolved job, split and hashes, fitted checkpoints, metrics, maps and mask overlays. Terminal output is also saved under `.workstation/logs/`. The **Re-run validation** and **Held-out test** options use the selected completed V5 run's original split, model state, and saved image threshold. Revalidation writes a dated subfolder, while the main benchmark report continues to use the original saved validation. A completed custom validation supports matched regular-light image AUROC/AP. The archived V1–V4 metrics retain their original capture protocols.

Use **Original V3/V4 benchmark** to run the existing benchmark against its original local data. It uses the benchmark hyperparameters. The inspection app continues to use saved V3 checkpoints. The report, Excel, journal, and publication actions read the saved run directories. Publication uses the existing shareable artifact selection. The datasets, raw maps, and local environments stay outside Git.

### PowerShell examples

```powershell
.\ANNAlyzer_Master.bat menu
.\ANNAlyzer_Master.bat setup --components unet reporting
.\ANNAlyzer_Master.bat new --config .workstation/jobs/workstation.json
.\ANNAlyzer_Master.bat render --config .workstation/jobs/workstation.json
.\ANNAlyzer_Master.bat audit --config .workstation/jobs/workstation.json
.\ANNAlyzer_Master.bat fit --config .workstation/jobs/workstation.json
.\ANNAlyzer_Master.bat html
```

Set `training.models` to the installed components before running the example fit. Start with a small render and one selected model to check the workflow on the new hardware before committing compute time to the full dataset.
