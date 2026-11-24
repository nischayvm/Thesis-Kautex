# ANNAlyzer AnomalyDINO baseline

The baseline uses Anomalib 2.5.1 with frozen DINOv2 ViT-S/14. It uses the existing local dataset and the frozen PatchCore split. The runner executes a one-tile smoke test before fitting red, green and regular separately. It reads the selected PatchCore results for comparison.

From this folder in PowerShell, create the isolated environment with the lockfile.

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.uv-cache'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.python'
uv sync --locked --python 3.12.13 --managed-python --link-mode copy --no-progress
```

Cache the pinned weights locally. The runner verifies their SHA-256 and runs offline.

```powershell
$revision = '4610ca143709d58a633b6397a74412c2c3842454'
$repo = '.hf-cache/hub/models--timm--vit_small_patch14_dinov2.lvd142m'
$snapshot = Join-Path $repo "snapshots/$revision"
New-Item -ItemType Directory -Force -Path $snapshot,(Join-Path $repo 'refs') | Out-Null
$weights = Join-Path $snapshot 'model.safetensors'
curl.exe --fail --location --output $weights "https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/resolve/$revision/model.safetensors"
if ($LASTEXITCODE -ne 0) { throw 'DINOv2 checkpoint download failed' }
if ((Get-FileHash $weights).Hash -ne '04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081') {
    throw 'DINOv2 checkpoint SHA-256 mismatch'
}
[System.IO.File]::WriteAllText((Join-Path (Resolve-Path (Join-Path $repo 'refs')) 'main'), $revision)
.\.venv\Scripts\python.exe -u run.py
```

The runner writes a new folder under `runs/` with fitted states, raw score maps, color-specific reports, visual overlays, a cross-model comparison CSV, and charts. Each tile keeps its 512 × 512 native pixels and receives a 3 px reflected border on every side. The 518 × 518 model map loses this border before Gaussian stitching. The model uses the stock last-layer features, 1-neighbor cosine distance and mean of the highest 1% of patch distances for the tile score. The frame score is the maximum tile score. Foreground masking and reference augmentation are disabled.

The benchmark selects 128 candidates from each tile using an 8 × 8 spatial partition of the 37 × 37 token grid. It then uses Anomalib's projected greedy k-center to retain 10%. Chunked cosine search keeps the same scoring rule as dense Anomalib inference. Metrics use the frozen PatchCore evaluator, and regular image-level metrics are N/A because the cohort has zero normal validation images.

[Anomalib 2.5.1 AnomalyDINO reference](https://anomalib.readthedocs.io/en/lib-v2.5.1/markdown/guides/reference/models/image/anomaly_dino.html) · [Pretrained DINOv2 checkpoint](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/blob/4610ca143709d58a633b6397a74412c2c3842454/model.safetensors) · [PatchCore freeze manifest](patchcore_freeze_v1.json)
