# ANNAlyzer PatchCore smoke test

From this folder in PowerShell, use the pinned environment and the official TorchVision checkpoint. The repository tracks the pretrained and fitted weights with Git LFS. Run `git lfs pull` if the clone contains weight pointers. The `curl` command below provides the official checkpoint separately when needed.

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.python'
uv sync --locked --python 3.12.13 --managed-python --no-progress
if (-not (Test-Path -LiteralPath 'wide_resnet50_2-9ba9bcbe.pth')) {
    curl.exe --fail --location --output wide_resnet50_2-9ba9bcbe.pth https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth
}
.\.venv\Scripts\python.exe -u smoke.py
```

The script validates the reference and three defective/mask pairs against the audit manifest by file hash. It reads the original dataset in place and writes a new directory under `runs/` for every run. The fitted state contains the frozen extractor and the normal-only coreset. `report.json` records the configuration, software versions, tensor dimensions, run timings, CUDA allocation and reserved-pool peaks, sampled host working-set peaks, and raw scores. The smoke cases span low, middle and high ground-truth mask coverage. They serve as qualitative diagnostics.

Each frame uses 25 native RGB tiles. Preprocessing scales RGB to `[0, 1]` and applies the checkpoint's ImageNet channel mean and standard deviation. Anomalib's TorchModel performs 3 × 3 average feature pooling, bilinear upsampling to the `layer2` grid, and feature concatenation. Patch scores use the exact Euclidean nearest neighbor. Anomalib upsamples each tile map with nearest-neighbor interpolation and Gaussian-smooths it with sigma 4 px. The nine-neighbor weighted tile scores are reported separately, with their maximum as the frame score. Map values are averaged where tiles overlap. Each `.npy` stores a raw FP32 map at 2048 × 2048. The heatmap PNGs share one display scale and the ground-truth overlays highlight mask pixels in magenta.

The setup uses the simple stock coreset only for the regular smoke test. For a future color run, first evaluate a seeded spatially stratified candidate cap, such as 64 candidates per tile across an 8 × 8 grid on the feature map. Each training image and every tile would contribute candidates. Stream those candidates to host memory and run Anomalib's projected k-center on the bounded pool, starting at 10% of that pool. Use the same policy and seed for both colors. Compare candidate budgets on validation only, including small-defect localization and actual coreset time, before fixing the method. This is a proposal for the next experiment.

Implementation references for the pinned version. [Anomalib 2.2 PatchCore TorchModel](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/patchcore/torch_model.py) · [Anomalib 2.2 anomaly map generator](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/patchcore/anomaly_map.py) · [TorchVision weight definition](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/resnet.py)
