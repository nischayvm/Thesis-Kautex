"""Run a fixed AnomalyDINO validation benchmark on the frozen ANNAlyzer split."""

import csv
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
PATCHCORE = HERE.parent / "patchcore"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PATCHCORE))
os.environ["HF_HOME"] = str(HERE / ".hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["MPLCONFIGDIR"] = str(HERE / ".mpl-cache")

import lighting_benchmark as reference
import phase3 as pipeline
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from anomalib.data import InferenceBatch
from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel
from matplotlib import colormaps, pyplot as plt
from PIL import Image, ImageDraw
from phase3_metrics import SIZE_LABELS
from smoke import FRAME, MIB, ORIGINS, OVERLAP, TILE, load_asset, measure, require, tile_tensor
from split_dataset import SEED, SPLIT, load_audit, split_text

ENCODER = "vit_small_patch14_dinov2.lvd142m"
REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
WEIGHT_HASH = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
WEIGHTS = HERE / ".hf-cache" / "hub" / f"models--timm--{ENCODER}" / "snapshots" / REVISION / "model.safetensors"
PATCH = 14
PAD = 3
PADDED = TILE + 2 * PAD
GRID = PADDED // PATCH
DIM = 384
SAMPLE_COUNT = 128
RATIO = 0.10
QUERY_CHUNK = 512
BANK_CHUNK = 8192
STITCH = "gaussian"
ORDER = ("red", "green", "regular")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BASELINE = PATCHCORE / "runs" / "lighting_benchmark_20261124T084630_908213Z" / "report.json"


class ChunkedAnomalyDINO(AnomalyDINOModel):
    """Reuse Anomalib's extractor, bank fit, top-1% score and map generator."""

    @staticmethod
    def padded(tile):
        require(tile.shape == (1, 3, TILE, TILE) and tile.dtype == torch.float32, "Expected one native FP32 RGB tile")
        return F.pad(tile, (PAD, PAD, PAD, PAD), mode="reflect")

    def forward(self, input_tensor):
        padded = self.padded(input_tensor)
        if self.training:
            return super().forward(padded)
        require(self.memory_bank.numel() > 0 and not self.masking and self.num_neighbours == 1, "Unexpected bank or scoring mode")
        features = self.extract_features(padded)
        require(features.shape == (1, GRID ** 2, DIM) and features.dtype == torch.float32, "Unexpected DINOv2 patch grid")
        features = F.normalize(features[0], p=2, dim=1)
        similarities = []
        for query in features.split(QUERY_CHUNK):
            best = torch.full((len(query),), -torch.inf, dtype=query.dtype, device=query.device)
            for chunk in self.memory_bank.split(BANK_CHUNK):
                best = torch.maximum(best, (query @ chunk.T).amax(dim=1))
            similarities.append(best)
        distances = (1 - torch.cat(similarities)).clamp_(0, 2).unsqueeze(0)
        tile_score = self.mean_top1p(distances)
        padded_map = self.anomaly_map_generator(distances.view(1, 1, GRID, GRID), (PADDED, PADDED))
        return InferenceBatch(pred_score=tile_score, anomaly_map=padded_map[:, :, PAD:PAD + TILE, PAD:PAD + TILE])


def make_model():
    model = ChunkedAnomalyDINO(num_neighbours=1, encoder_name=ENCODER, masking=False,
                               coreset_subsampling=True, sampling_ratio=RATIO)
    model.requires_grad_(False)
    require(model.patch_size == PATCH and not any(p.requires_grad for p in model.parameters()), "Backbone freeze or patch size failed")
    return model.float().cuda().eval()


def stratified_indices(rng):
    edges = np.linspace(0, GRID, 9, dtype=np.int32)
    first, second = [], []
    for y0, y1 in zip(edges[:-1], edges[1:], strict=True):
        for x0, x1 in zip(edges[:-1], edges[1:], strict=True):
            width = x1 - x0
            size = (y1 - y0) * width
            a = rng.integers(size)
            b = (a + rng.integers(1, size)) % size
            first.append((y0 + a // width) * GRID + x0 + a % width)
            second.append((y0 + b // width) * GRID + x0 + b % width)
    return first + second


def collect(model, records, mean, std, origins):
    rng = np.random.default_rng(SEED)
    count = len(records) * len(origins) * SAMPLE_COUNT // 2
    candidates = torch.empty((2 * count, DIM), dtype=torch.float32, device="cpu")
    model.train()
    model.feature_encoder.eval()
    offset = 0
    for number, record in enumerate(records, 1):
        asset = record["files"]["normal"]
        require(asset["label"] == "normal", "Only audited normal images enter the bank")
        rgb = load_asset(asset, "RGB")
        for y, x in origins:
            model(tile_tensor(rgb, y, x, mean, std))
            features = model.embedding_store.pop()
            require(features.shape == (GRID ** 2, DIM) and features.dtype == torch.float32, "Unexpected normal patch features")
            indices = torch.as_tensor(stratified_indices(rng), device=features.device, dtype=torch.long)
            chosen = features[indices].cpu()
            candidates[offset:offset + 64].copy_(chosen[:64])
            candidates[count + offset:count + offset + 64].copy_(chosen[64:])
            offset += 64
        if number % 10 == 0:
            print(f"Collected {number}/{len(records)} normal frames for {record['lighting']}", flush=True)
    require(2 * offset == len(candidates) and not model.embedding_store, "Incorrect bounded candidate count")
    return candidates


def fit(model, candidates):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    require(all(torch.isfinite(part).all().item() for part in candidates.split(BANK_CHUNK)), "Nonfinite candidates")
    model.embedding_store.append(candidates.cuda())
    model.fit()
    model.eval()
    expected = (int(len(candidates) * RATIO), DIM)
    require(model.memory_bank.shape == expected and model.memory_bank.dtype == torch.float32
            and torch.isfinite(model.memory_bank).all().item(), f"Invalid fitted bank {model.memory_bank.shape}")


def smoke_test(out, train, validation, mean, std):
    print("Running one-tile, normal-only CUDA smoke test", flush=True)
    model = make_model()
    y, x = ORIGINS[len(ORIGINS) // 2]
    rgb = load_asset(validation["files"]["bubbles"], "RGB")
    with torch.inference_mode():
        fit(model, collect(model, [train], mean, std, [(y, x)]))
        tile = tile_tensor(rgb, y, x, mean, std)
        padded = model.padded(tile)
        require(padded.shape == (1, 3, PADDED, PADDED) and torch.equal(padded[:, :, PAD:PAD + TILE, PAD:PAD + TILE], tile),
                "Padding changed native pixels")
        features = model.extract_features(padded)
        require(features.shape == (1, GRID ** 2, DIM), "Unexpected pretrained grid")
        scored = model(tile)
        stock = AnomalyDINOModel.forward(model, padded)
        dense_cropped = stock.anomaly_map[:, :, PAD:PAD + TILE, PAD:PAD + TILE]
        torch.testing.assert_close(scored.anomaly_map, dense_cropped, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(scored.pred_score, stock.pred_score, atol=2e-6, rtol=1e-5)
        torch.save(model.state_dict(), out / "smoke_state.pt")
        reloaded = make_model()
        reloaded.load_state_dict(torch.load(out / "smoke_state.pt", map_location="cpu", weights_only=True))
        restored = reloaded(tile)
        torch.testing.assert_close(restored.anomaly_map, scored.anomaly_map, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(restored.pred_score, scored.pred_score, atol=1e-6, rtol=1e-6)
        require(torch.isfinite(scored.anomaly_map).all().item(), "Nonfinite smoke result")
        return {"fit_id": train["sample_id"], "validation_id": validation["sample_id"], "origin_yx": [y, x],
                "native_input": list(tile.shape), "padded_input": list(padded.shape), "features": list(features.shape),
                "bank": list(model.memory_bank.shape), "map": list(scored.anomaly_map.shape),
                "dense_comparison_max_error": (scored.anomaly_map - dense_cropped).abs().max().item(),
                "reload_max_error": (restored.anomaly_map - scored.anomaly_map).abs().max().item()}


def run_colour(out, colour, train, validation, stitcher, mean, std):
    folder = out / colour
    (folder / "raw").mkdir(parents=True, exist_ok=False)
    timings = {}
    with measure(f"{colour}_load_encoder", timings):
        model = make_model()
    with torch.inference_mode():
        with measure(f"{colour}_candidate_extraction", timings):
            candidates = collect(model, train, mean, std, ORIGINS)
        candidate_count = len(candidates)
        with measure(f"{colour}_coreset", timings):
            fit(model, candidates)
        del candidates
        checkpoint = reference.save_model(model, folder / "fitted_state.pt")
        torch.cuda.empty_cache()
        with measure(f"{colour}_validation_inference", timings):
            predictions = pipeline.evaluate(model, validation, SAMPLE_COUNT, stitcher, folder, mean, std,
                                            stitches=(STITCH,), include_normal=(colour != "regular"))
    del model
    torch.cuda.empty_cache()
    start = perf_counter()
    metrics, metric_score_max = reference.score_validation(folder, predictions, validation)
    result = {"train_ids": [r["sample_id"] for r in train], "validation_ids": [r["sample_id"] for r in validation],
              "validation_normal_count": sum(row["role"] == "normal" for row in predictions),
              "candidate_rows": candidate_count, "checkpoint": checkpoint, "measurements": timings,
              "metric_score_max": metric_score_max, "metrics": metrics, "predictions": predictions,
              "metrics_seconds": perf_counter() - start}
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def save_visuals(out, results, cohorts, examples):
    colour_map = colormaps["inferno"]
    maximum = max(p["maps"][STITCH]["max"] for result in results.values() for p in result["predictions"])
    sheet = Image.new("RGB", (1656, 928), "#1a1a1a")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"AnomalyDINO validation | shared display scale: 0 to {maximum:.4f}", fill="white")
    for row_index, colour in enumerate(reference.COLOURS):
        folder = out / colour / "visuals"
        folder.mkdir()
        validation = {record["sample_id"]: record for record in cohorts[colour][1]}
        for pred in results[colour]["predictions"]:
            if pred["role"] != "bubbles":
                continue
            key = pred["sample_id"]
            sample = key.split("/")[-1]
            record = validation[key]
            _, foreground = reference.defect_targets(record)
            rgb = load_asset(record["files"]["bubbles"], "RGB")
            rgb[foreground] = np.round(0.45 * rgb[foreground] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
            score = np.load(folder.parent / pred["maps"][STITCH]["path"], allow_pickle=False)
            overlay = Image.fromarray(rgb)
            heatmap = Image.fromarray(colour_map(np.clip(score / maximum, 0, 1), bytes=True)[..., :3])
            overlay.save(folder / f"{sample}_ground_truth_overlay.png")
            heatmap.save(folder / f"{sample}_heatmap.png")
            if sample in examples[colour]:
                index = examples[colour].index(sample)
                y = 35 + row_index * 294
                for offset, (name, image) in enumerate((("ground truth", overlay), ("heatmap", heatmap))):
                    x = 8 + (2 * index + offset) * 275
                    draw.text((x, y), f"{key} · {name}", fill="white")
                    sheet.paste(image.resize((265, 265), Image.Resampling.LANCZOS), (x, y + 18))
    sheet.save(out / "lighting_heatmaps.png")
    return [0, maximum]


def save_model_comparison(out, results, baseline):
    fields = ("pixel_auroc", "pixel_ap", "aupro", "image_auroc", "image_ap")
    names = ("Pixel AUROC", "Pixel AP", "AUPRO at 30% FPR", "Image AUROC", "Image AP")
    models = (("PatchCore", baseline, "#4a628b"), ("AnomalyDINO", results, "#ad663c"))
    with (out / "model_comparison.csv").open("x", encoding="utf-8", newline="") as output:
        table = csv.writer(output)
        table.writerow(("model", "lighting", *fields))
        for model, conditions, _ in models:
            for colour in reference.COLOURS:
                metrics = conditions[colour]["metrics"]
                table.writerow((model, colour, *((metrics[k] if metrics[k] is not None else "N/A") for k in fields)))
    fig, axes = plt.subplots(1, len(fields), figsize=(14.5, 3.8), sharey=True, layout="constrained")
    for ax, key, title in zip(axes, fields, names, strict=True):
        for i, (name, conditions, shade) in enumerate(models):
            for x, colour in enumerate(reference.COLOURS):
                at = x + (i - .5) * .34
                value = conditions[colour]["metrics"][key]
                if value is None:
                    ax.text(at, .03, "N/A", ha="center", fontsize=6, color=shade)
                else:
                    ax.bar(at, value, width=.30, color=shade, label=name if x == 1 else None)
                    ax.text(at, value + .02, f"{value:.3f}", ha="center", fontsize=6)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(3), reference.LABELS, fontsize=8)
        ax.set_ylim(0, 1.10)
        ax.set_xlim(-.5, 2.5)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    fig.savefig(out / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True, layout="constrained")
    for ax, colour, label in zip(axes, reference.COLOURS, reference.LABELS, strict=True):
        for index, (name, conditions, shade) in enumerate(models):
            sizes = conditions[colour]["metrics"]["size_specific"]
            values = [sizes[key]["aupro"] for key in SIZE_LABELS]
            ax.bar(np.arange(3) + (index - .5) * .34, values, width=.30, color=shade, label=name)
        ax.set_title(label)
        ax.set_xticks(range(3), ("<64", "64–511", "≥512"))
        ax.set_xlabel("Visible component area (px²)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=2, frameon=False)
    axes[0].set_ylabel("AUPRO at 30% FPR")
    fig.savefig(out / "defect_size_comparison.png", dpi=180)
    plt.close(fig)


def main():
    cv2.setNumThreads(2)
    require(torch.cuda.is_available() and torch.version.cuda == "12.8", "Expected CUDA 12.8")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    records, audit_sha = load_audit()
    split_hash = reference.sha256(SPLIT)
    split_contents = SPLIT.read_text(encoding="utf-8")
    require(split_contents == split_text(records, audit_sha), "Frozen split differs from its audit assignment")
    split = json.loads(split_contents)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_hash = reference.sha256(BASELINE)
    require(baseline["configuration"]["split_sha256"] == split_hash, "PatchCore split differs")
    require(reference.sha256(WEIGHTS) == WEIGHT_HASH, "Unexpected DINOv2 pretrained weights")
    cohorts = {colour: reference.cohort(records, split, colour) for colour in ORDER}
    for colour, (train, validation) in cohorts.items():
        frozen = baseline["lighting"][colour]
        require([r["sample_id"] for r in train] == frozen["train_ids"]
                and [r["sample_id"] for r in validation] == frozen["validation_ids"], "Cohort changed")

    out = HERE / "runs" / datetime.now(timezone.utc).strftime("anomalydino_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SPLIT, out / SPLIT.name)
    print(f"Run directory {out}", flush=True)
    mean = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)
    smoke = smoke_test(out, cohorts["red"][0][0], cohorts["red"][1][0], mean, std)
    (out / "smoke.json").write_text(json.dumps(smoke, indent=2) + "\n", encoding="utf-8")
    print("Smoke test passed: native padding, upstream score parity and checkpoint reload verified", flush=True)
    torch.cuda.empty_cache()
    stitcher = pipeline.Stitcher()
    results = {colour: run_colour(out, colour, *cohorts[colour], stitcher, mean, std) for colour in ORDER}
    save_model_comparison(out, results, baseline["lighting"])
    red_examples = ["0008", "0041", "0042"]
    display = save_visuals(out, results, cohorts, {"red": red_examples, "green": red_examples,
                            "regular": [r["sample_id"].split("/")[-1] for r in cohorts["regular"][1]]})
    require(reference.sha256(SPLIT) == split_hash and reference.sha256(BASELINE) == baseline_hash, "Frozen input changed")
    report = {
        "method": "Anomalib AnomalyDINO",
        "configuration": {"encoder": ENCODER, "encoder_revision": REVISION, "encoder_sha256": WEIGHT_HASH,
                          "seed": SEED, "precision": "FP32; TF32 disabled", "backbone_frozen": True, "batch_size": 1,
                          "reference_augmentation": False, "foreground_masking": False, "rgb_mean": MEAN, "rgb_std": STD,
                          "tile": TILE, "tile_origins_yx": ORIGINS, "overlap": OVERLAP, "padded_tile": PADDED,
                          "reflect_padding_pixels_per_side": PAD, "resize": None, "input_crop": None,
                          "patch_size": PATCH, "patch_grid": [GRID, GRID], "feature_dim": DIM,
                          "candidates_per_tile": SAMPLE_COUNT, "coreset_ratio": RATIO,
                          "sampler": "two distinct seeded tokens per cell of an 8 by 8 spatial partition",
                          "nearest_neighbors": 1, "distance": "cosine", "query_chunk": QUERY_CHUNK, "bank_chunk": BANK_CHUNK,
                          "image_aggregation": "maximum over tile means of the largest 1% of token distances",
                          "stitching": STITCH, "gaussian_sigma_px": pipeline.SIGMA, "threshold": None,
                          "split_sha256": split_hash, "patchcore_report_sha256": baseline_hash},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "torchvision": version("torchvision"), "anomalib": version("anomalib"), "timm": version("timm"),
                        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "runner_sha256": reference.sha256(Path(__file__)), "smoke": smoke, "shared_heatmap_display_scale": display,
        "lighting": results,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved consolidated AnomalyDINO results to {out}", flush=True)


if __name__ == "__main__":
    main()
