# ANNAlyzer supervised U-Net

The benchmark trains separate red and green U-Nets with ImageNet-pretrained ResNet18 encoders. Each fit uses all 25 native tiles from each of its 70 normal and 70 defective training frames. Normal frames have zero masks. The loss adds BCE with logits and soft Dice loss with smoothing 1. The decoder widths from coarse to fine are 128, 64, 32, 16, and 8.

The fixed settings are AdamW with learning rate `1e-4` and weight decay `0.01`, 10 epochs, batch size 4, CUDA FP16 autocast with gradient scaling, and seed 24117. Both the encoder and decoder are fitted. Each epoch uses the frozen in-lighting validation split to select the highest pixel AP on stitched native-resolution maps. The earliest epoch wins ties. Regular evaluation transfers the selected red checkpoint to the three existing regular development defects.

From this directory in PowerShell, prepare the isolated environment and pinned encoder weights.

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.python'
uv sync --locked --python 3.12.13 --managed-python --link-mode copy --no-progress
curl.exe --fail --location --output resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
.\.venv\Scripts\python.exe -u run.py
```

The runner reads the audited local dataset under `data/annalyzer_batch_2048` and prints the new run directory. The latest epoch checkpoint supports restart using `run.py --resume <run-directory>`. After training, run `render_results.py <run-directory>` with the same interpreter to export the three-model table, charts, probability maps and overlays. The charts use stored validation reports from PatchCore and AnomalyDINO. The shared evaluation preserves 512 × 512 tiles, 128 px overlap, sigma 128 px Gaussian weights and native 2048 × 2048 stitched output.

[TorchVision ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) · [AnomalyDINO freeze manifest](anomalydino_freeze_v1.json) · [PatchCore freeze manifest](../anomalydino/patchcore_freeze_v1.json)
