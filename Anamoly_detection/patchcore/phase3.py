"""Fit two red-lighting PatchCore banks and compare stitching on validation."""

import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import smoke
import cv2
import numpy as np
import psutil
import torch
import torchvision
from anomalib.models.components import KCenterGreedy
from matplotlib import colormaps
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score

from phase3_metrics import BINS, DIAGNOSTIC_FPR, FPR_LIMIT, PixelMetrics, RegionTargets
from smoke import FRAME, HERE, MIB, ORIGINS, OVERLAP, RATIO, STRIDE, TILE, WEIGHTS, load_asset, measure, require, tile_tensor
from split_dataset import SEED, SPLIT, load_audit, split_text

BUDGETS = (64, 128)
STITCHES = ("uniform", "gaussian")
SIGMA = TILE / 4
GRID = 64
DIMENSION = 1536
SEAM_CONTROL_OFFSET = 16


def sampled_indices(rng):
    """Take one or two distinct cells from each 8×8 spatial stratum."""
    strata = np.arange(64)
    first = rng.integers(64, size=64)
    second = (first + rng.integers(1, 64, size=64)) % 64
    offsets = np.stack((first, second))
    rows = 8 * (strata // 8) + offsets // 8
    cols = 8 * (strata % 8) + offsets % 8
    return (rows * GRID + cols).ravel()


def collect_candidates(model, training, mean, std):
    count = len(training) * len(ORIGINS)
    primary = torch.empty((count * 64, DIMENSION), dtype=torch.float32, device="cpu")
    secondary = torch.empty_like(primary)
    rng = np.random.default_rng(SEED)
    model.train()
    model.feature_extractor.eval()
    offset = 0
    for number, record in enumerate(training, 1):
        normal = record["files"]["normal"]
        require(normal["label"] == "normal", f"Unexpected training asset {normal['path']}")
        image = load_asset(normal, "RGB")
        for y, x in ORIGINS:
            model(tile_tensor(image, y, x, mean, std))
            embedding = model.embedding_store.pop()
            require(embedding.shape == (GRID ** 2, DIMENSION) and embedding.dtype == torch.float32,
                    f"Unexpected tile embedding {embedding.shape}, {embedding.dtype}")
            chosen = embedding[torch.as_tensor(sampled_indices(rng), device=embedding.device)].cpu()
            primary[offset:offset + 64].copy_(chosen[:64])
            secondary[offset:offset + 64].copy_(chosen[64:])
            offset += 64
        if number % 10 == 0:
            print(f"Extracted {number}/{len(training)} red training views", flush=True)
    require(offset == count * 64 and not model.embedding_store, "Incorrect candidate count")
    model.eval()
    return primary, secondary


def fit_bank(primary, secondary, budget):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    count = len(primary) * (budget // 64)
    candidates = torch.empty((count, DIMENSION), dtype=torch.float32, device="cuda")
    candidates[:len(primary)].copy_(primary)
    if budget == 128:
        candidates[len(primary):].copy_(secondary)
    require(torch.isfinite(candidates).all().item(), "Nonfinite training candidates")
    bank = KCenterGreedy(embedding=candidates, sampling_ratio=RATIO).sample_coreset()
    require(bank.shape == (int(count * RATIO), DIMENSION) and torch.isfinite(bank).all().item(), "Invalid coreset")
    return bank


class Stitcher:
    def __init__(self):
        centered = np.arange(TILE, dtype=np.float32) - (TILE - 1) / 2
        gaussian = np.exp(-0.5 * (centered / SIGMA) ** 2)
        self.weights = {"uniform": np.ones((TILE, TILE), dtype=np.float32), "gaussian": np.outer(gaussian, gaussian)}
        self.denominators = {name: np.zeros((FRAME, FRAME), dtype=np.float32) for name in STITCHES}
        for y, x in ORIGINS:
            for name, weights in self.weights.items():
                self.denominators[name][y:y + TILE, x:x + TILE] += weights
        require(self.denominators["uniform"].min() == 1 and self.denominators["uniform"].max() == 4
                and self.denominators["gaussian"].min() > 0, "Invalid stitching coverage")

    def apply(self, tile_maps, name):
        result = np.zeros((FRAME, FRAME), dtype=np.float32)
        for (y, x), tile in zip(ORIGINS, tile_maps, strict=True):
            result[y:y + TILE, x:x + TILE] += tile * self.weights[name]
        result /= self.denominators[name]
        return result


def infer_tiles(model, rgb, mean, std):
    maps, scores = [], []
    for y, x in ORIGINS:
        prediction = model(tile_tensor(rgb, y, x, mean, std))
        require(prediction.anomaly_map.shape == (1, 1, TILE, TILE)
                and prediction.anomaly_map.dtype == torch.float32, "Invalid tile map")
        maps.append(prediction.anomaly_map[0, 0].cpu().numpy())
        scores.append(prediction.pred_score.item())
    require(np.isfinite(scores).all(), "Nonfinite tile score")
    return maps, max(scores)


def seam_steps(score):
    axis_origins = sorted({x for _, x in ORIGINS})
    edges = sorted({line for start in axis_origins for line in (start, start + TILE) if 0 < line < FRAME})

    def mean_step(lines):
        return float(np.mean([np.abs(view[:, x] - view[:, x - 1]).mean()
                              for x in lines for view in (score, score.T)]))

    controls = [edge + offset for edge in edges for offset in (-SEAM_CONTROL_OFFSET, SEAM_CONTROL_OFFSET)]
    return mean_step(edges), mean_step(controls)


def evaluate(model, validation, budget, stitcher, out, mean, std):
    results = []
    for index, record in enumerate(validation, 1):
        sample = record["sample_id"].split("/")[-1]
        for role in ("normal", "bubbles"):
            image = load_asset(record["files"][role], "RGB")
            start = perf_counter()
            tiles, score = infer_tiles(model, image, mean, std)
            row = {"sample_id": record["sample_id"], "role": role, "score": score,
                   "tile_inference_s": perf_counter() - start, "maps": {}}
            for stitch in STITCHES:
                start = perf_counter()
                raw = stitcher.apply(tiles, stitch)
                elapsed = perf_counter() - start
                require(np.isfinite(raw).all() and raw.min() >= 0, "Nonfinite or negative stitched map")
                name = f"raw/{budget}_{stitch}_{sample}_{role}.npy"
                np.save(out / name, raw, allow_pickle=False)
                row["maps"][stitch] = {"path": name, "max": float(raw.max()), "stitch_s": elapsed}
                if role == "normal":
                    seam, control = seam_steps(raw)
                    row["maps"][stitch].update(seam_step=seam, control_step=control)
            results.append(row)
        if index % 5 == 0:
            print(f"{budget} candidates: scored {index}/{len(validation)} red validation pairs", flush=True)
    return results


def summarize(out, validation, results):
    maximum = max(entry["max"] for rows in results.values() for row in rows for entry in row["maps"].values())
    metrics = {(budget, stitch): PixelMetrics(maximum) for budget in BUDGETS for stitch in STITCHES}
    visual_path = out / "visuals"
    visual_path.mkdir()
    display = colormaps["inferno"]
    checked = sorted(validation, key=lambda row: row["mask_positive_pixels"])
    representative = {checked[index]["sample_id"] for index in (0, len(checked) // 2, -1)}
    preview_rows = []
    for index, record in enumerate(validation):
        sample = record["sample_id"].split("/")[-1]
        mask = load_asset(record["files"]["masks"], "L")
        foreground = mask == 255
        require(np.all((mask == 0) | foreground) and foreground.sum() == record["mask_positive_pixels"],
                f"Invalid ground truth {record['sample_id']}")
        regions = RegionTargets(foreground)
        background = RegionTargets(np.zeros_like(foreground))
        rgb = load_asset(record["files"]["bubbles"], "RGB")
        overlay = rgb.copy()
        overlay[foreground] = np.round(0.45 * overlay[foreground] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
        overlay_file = visual_path / f"{sample}_ground_truth_overlay.png"
        Image.fromarray(overlay).save(overlay_file)
        Image.fromarray(mask).save(visual_path / f"{sample}_mask.png")
        preview = [overlay_file]
        for (budget, stitch), metric in metrics.items():
            for offset, targets in ((0, background), (1, regions)):
                row = results[budget][2 * index + offset]
                require(row["sample_id"] == record["sample_id"], "Misordered validation predictions")
                score = np.load(out / row["maps"][stitch]["path"], allow_pickle=False)
                metric.update(score, targets)
                if offset == 1:
                    pixels = display(np.clip(score / maximum, 0, 1), bytes=True)[..., :3]
                    file = visual_path / f"{sample}_{budget}_{stitch}_heatmap.png"
                    Image.fromarray(pixels).save(file)
                    preview.append(file)
        if record["sample_id"] in representative:
            preview_rows.append((record["sample_id"], preview))

    titles = ["Ground truth", "64 uniform", "64 Gaussian", "128 uniform", "128 Gaussian"]
    canvas = Image.new("RGB", (1500, 946), "#151515")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(titles):
        draw.text((column * 300 + 4, 8), title, fill="white")
    for row, (label, files) in enumerate(preview_rows):
        y = 28 + row * 305
        draw.text((4, y), label, fill="white")
        for column, file in enumerate(files):
            with Image.open(file) as image:
                canvas.paste(image.resize((292, 292), Image.Resampling.LANCZOS), (column * 300 + 4, y + 14))
    canvas.save(out / "comparison.png")

    summary = {}
    for (budget, stitch), metric in metrics.items():
        row = metric.compute()
        coarse = metric.compute(coarsen=2)
        row["half_bin_resolution_abs_difference"] = {key: abs(row[key] - coarse[key])
                                                       for key in ("pixel_auroc", "pixel_ap", "aupro")}
        background = [result["maps"][stitch] for result in results[budget] if result["role"] == "normal"]
        seam = float(np.mean([entry["seam_step"] for entry in background]))
        control = float(np.mean([entry["control_step"] for entry in background]))
        row["normal_seam"] = {"step": seam, "control_step": control, "ratio": seam / control}
        row["mean_inference_s"] = float(np.mean([x["tile_inference_s"] + x["maps"][stitch]["stitch_s"]
                                               for x in results[budget]]))
        labels = [x["role"] == "bubbles" for x in results[budget]]
        scores = [x["score"] for x in results[budget]]
        row["image_auroc"] = float(roc_auc_score(labels, scores))
        row["image_ap"] = float(average_precision_score(labels, scores))
        summary[f"{budget}_{stitch}"] = row
    return summary, maximum, [label for label, _ in preview_rows]


def main():
    cv2.setNumThreads(2)
    require(torch.cuda.is_available() and torch.version.cuda == "12.8", "Expected the Phase 2 CUDA environment")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    records, audit_digest = load_audit()
    expected_split = split_text(records, audit_digest)
    require(SPLIT.read_text(encoding="utf-8") == expected_split, "Saved split does not match the deterministic assignment")
    folds = json.loads(expected_split)["coloured_setup_ids"]
    red = {name: [records[f"red/{key}"] for key in folds[name]] for name in ("train", "validation")}
    with WEIGHTS.open("rb") as source:
        weight_sha = hashlib.file_digest(source, "sha256").hexdigest()
    require(weight_sha == "9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950", "Backbone hash changed")

    out = HERE / "runs" / datetime.now(timezone.utc).strftime("red_phase3_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir()
    shutil.copyfile(SPLIT, out / SPLIT.name)
    print(f"Run directory {out}", flush=True)
    measurements, results, sizes = {}, {}, {}
    preset = smoke.Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms()
    mean = torch.tensor(preset.mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(preset.std, device="cuda").view(1, 3, 1, 1)
    stitcher = Stitcher()
    with measure("load_backbone", measurements):
        model = smoke.make_model(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        with measure("shared_candidate_extraction", measurements):
            primary, secondary = collect_candidates(model, red["train"], mean, std)
        for budget in BUDGETS:
            torch.cuda.empty_cache()
            with measure(f"coreset_{budget}", measurements):
                model.memory_bank = fit_bank(primary, secondary, budget)
            sizes[budget] = {"candidate_rows": len(primary) * (budget // 64), "dimensions": list(model.memory_bank.shape),
                             "bank_mib": model.memory_bank.nbytes / MIB}
            torch.save(model.memory_bank.cpu(), out / f"memory_bank_{budget}.pt")
            torch.cuda.empty_cache()
            with measure(f"validation_inference_{budget}", measurements):
                results[budget] = evaluate(model, red["validation"], budget, stitcher, out, mean, std)
            model.memory_bank = torch.empty(0, device="cuda")
    del primary, secondary, model
    torch.cuda.empty_cache()
    with measure("validation_metrics_and_visuals", measurements):
        metrics, display_max, previews = summarize(out, red["validation"], results)

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=False)
    report = {
        "configuration": {
            "split_sha256": hashlib.sha256(SPLIT.read_bytes()).hexdigest(), "git_base": git.stdout.strip(),
            "script_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                              for name in ("phase3.py", "phase3_metrics.py", "smoke.py", "split_dataset.py")},
            "seed": SEED, "fit_colour": "red", "fit_normal_ids": folds["train"], "scored_validation_ids": folds["validation"],
            "backbone": "wide_resnet50_2.IMAGENET1K_V2", "weight_sha256": weight_sha, "layers": smoke.LAYERS,
            "rgb_mean": preset.mean, "rgb_std": preset.std, "precision": "FP32; TF32 disabled", "batch_size": 1,
            "tile": TILE, "overlap": OVERLAP, "stride": STRIDE, "resize": None, "crop": None, "coreset_ratio": RATIO,
            "sampler": "nested one/two distinct samples per cell of the 8 x 8 grid on a 64 x 64 feature map",
            "coreset": "Anomalib sparse random projection and greedy k-center", "nearest_neighbors": smoke.NEIGHBORS,
            "query_chunk": smoke.QUERY_CHUNK, "bank_chunk": smoke.BANK_CHUNK, "stitch": STITCHES,
            "gaussian_sigma_px": SIGMA, "seam_control_offset_px": SEAM_CONTROL_OFFSET,
            "display_range": [0, display_max], "metric_histogram_bins": BINS, "aupro_fpr_limit": FPR_LIMIT,
            "region_diagnostic_fpr": DIAGNOSTIC_FPR, "display_examples": previews,
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "torchvision": torchvision.__version__, "anomalib": version("anomalib"),
                        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / MIB,
                        "system_ram_mib": psutil.virtual_memory().total / MIB},
        "banks": sizes, "measurements": measurements, "metrics": metrics, "per_image": results,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validation results to {out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
