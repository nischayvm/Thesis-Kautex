"""Fit and verify a single-reference ANNAlyzer PatchCore smoke test."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))
os.environ.setdefault("TQDM_MININTERVAL", "5")

import hashlib
import io
import json
import math
import platform
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from threading import Event, Thread
from time import perf_counter

import numpy as np
import psutil
import torch
import torchvision
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from matplotlib import colormaps
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

DATASET = HERE.parents[1] / "data" / "annalyzer_batch_2048"
MANIFEST = HERE.parent / "dataset_audit_manifest_v1.jsonl"
WEIGHTS = HERE / "wide_resnet50_2-9ba9bcbe.pth"
FRAME = 2048
TILE = 512
OVERLAP = 128
STRIDE = TILE - OVERLAP
ORIGINS = [(y, x) for y in range(0, FRAME - TILE + 1, STRIDE)
           for x in range(0, FRAME - TILE + 1, STRIDE)]
LAYERS = ("layer2", "layer3")
RATIO = 0.1
NEIGHBORS = 9
QUERY_CHUNK = 512
BANK_CHUNK = 8_192
SEED = 24117
SAMPLES = ("regular/0004", "regular/0074", "regular/0097")
MIB = 1024 ** 2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class ChunkedPatchCore(PatchcoreModel):
    """Keep Anomalib's exact Euclidean search and bound each distance matrix."""

    def nearest_neighbors(self, embedding, n_neighbors):
        values, indices = [], []
        for query in embedding.split(QUERY_CHUNK):
            best_values = best_indices = None
            for offset in range(0, len(self.memory_bank), BANK_CHUNK):
                bank = self.memory_bank[offset:offset + BANK_CHUNK]
                distances = self.euclidean_dist(query, bank)
                k = min(n_neighbors, len(bank))
                if n_neighbors == 1:
                    block_values, block_indices = distances.min(dim=1, keepdim=True)
                else:
                    block_values, block_indices = distances.topk(k, dim=1, largest=False)
                block_indices += offset
                if best_values is None:
                    best_values, best_indices = block_values, block_indices
                elif n_neighbors == 1:
                    better = block_values < best_values
                    best_values = torch.where(better, block_values, best_values)
                    best_indices = torch.where(better, block_indices, best_indices)
                else:
                    candidates = torch.cat((best_values, block_values), dim=1)
                    best_values, order = candidates.topk(n_neighbors, dim=1, largest=False)
                    best_indices = torch.cat((best_indices, block_indices), dim=1).gather(1, order)
            values.append(best_values.squeeze(1) if n_neighbors == 1 else best_values)
            indices.append(best_indices.squeeze(1) if n_neighbors == 1 else best_indices)
        return torch.cat(values), torch.cat(indices)


@contextmanager
def measure(name, measurements):
    process = psutil.Process()
    stop = Event()
    peak = process.memory_info().rss

    def poll_rss():
        nonlocal peak
        while not stop.wait(0.05):
            peak = max(peak, process.memory_info().rss)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    sampler = Thread(target=poll_rss, daemon=True)
    sampler.start()
    try:
        yield
    finally:
        stop.set()
        sampler.join()
        torch.cuda.synchronize()
        row = {
            "seconds": round(perf_counter() - start, 3),
            "rss_peak_mib_sampled": round(max(peak, process.memory_info().rss) / MIB, 1),
            "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 1),
            "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 1),
        }
        measurements[name] = row
        print(name, json.dumps(row), flush=True)


def load_asset(asset, mode):
    data = (DATASET / asset["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == asset["sha256"], f"Audit hash mismatch for {asset['path']}")
    with Image.open(io.BytesIO(data)) as image:
        require(image.format == "PNG" and image.mode == mode and image.size == (FRAME, FRAME), asset["path"])
        return np.array(image)


def make_model(state=None):
    backbone = wide_resnet50_2(weights=None)
    if state is not None:
        backbone.load_state_dict(state)
    backbone.requires_grad_(False)
    model = ChunkedPatchCore(backbone=backbone.eval(), layers=LAYERS, pre_trained=False, num_neighbors=NEIGHBORS)
    require(model.tiler is None, "Unexpected internal tiler")
    return model.float().cuda()


def tile_tensor(rgb, y, x, mean, std):
    tile = np.ascontiguousarray(rgb[y:y + TILE, x:x + TILE])
    tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to("cuda", dtype=torch.float32)
    require(tensor.shape == (1, 3, TILE, TILE), f"Unexpected tile shape {tensor.shape}")
    return (tensor.div_(255.0) - mean) / std


def infer_frame(model, rgb, mean, std):
    score_sum = np.zeros((FRAME, FRAME), dtype=np.float32)
    coverage = np.zeros((FRAME, FRAME), dtype=np.uint8)
    tile_scores = []
    for y, x in ORIGINS:
        prediction = model(tile_tensor(rgb, y, x, mean, std))
        require(prediction.anomaly_map.shape == (1, 1, TILE, TILE), "Unexpected anomaly map shape")
        require(prediction.anomaly_map.dtype == torch.float32 and prediction.pred_score.shape == (1,),
                "Unexpected map precision or tile score shape")
        tile_map = prediction.anomaly_map[0, 0].cpu().numpy()
        score = prediction.pred_score.item()
        require(np.isfinite(tile_map).all() and np.isfinite(score), "Nonfinite inference output")
        score_sum[y:y + TILE, x:x + TILE] += tile_map
        coverage[y:y + TILE, x:x + TILE] += 1
        tile_scores.append(score)
    require(coverage.min() == 1 and coverage.max() == 4, "Incorrect tile coverage")
    score_sum /= coverage
    return score_sum, tile_scores


def capacity_estimates(rows_per_tile, dimensions):
    estimates = {}
    for image_count in (70, 100):
        rows = image_count * len(ORIGINS) * rows_per_tile
        coreset_rows = int(rows * RATIO)
        gib = 1024 ** 3
        projected_dimensions = int(4 * math.log(rows) / (0.9 ** 2 / 2 - 0.9 ** 3 / 3))
        estimates[str(image_count)] = {
            "per_color": True,
            "tiles": image_count * len(ORIGINS),
            "candidate_rows": rows,
            "candidate_gib": rows * dimensions * 4 / gib,
            "stock_vstack_lower_bound_gib": 2 * rows * dimensions * 4 / gib,
            "stock_projected_matrix_gib": rows * projected_dimensions * 4 / gib,
            "coreset_rows": coreset_rows,
            "coreset_gib": coreset_rows * dimensions * 4 / gib,
            "unchunked_distance_gib_per_tile": rows_per_tile * coreset_rows * 4 / gib,
            "bounded_distance_block_mib": QUERY_CHUNK * BANK_CHUNK * 4 / MIB,
            "sampled_candidates_64_per_tile_mib": image_count * len(ORIGINS) * 64 * dimensions * 4 / MIB,
        }
    return estimates


def main():
    require(torch.cuda.is_available(), "CUDA unavailable in this Python environment")
    require(torch.version.cuda == "12.8", f"Unexpected CUDA build {torch.version.cuda}")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    cuda_probe = (torch.ones(32, device="cuda") @ torch.ones(32, device="cuda")).item()
    require(cuda_probe == 32, "Unexpected CUDA probe result")
    props = torch.cuda.get_device_properties(0)
    cuda_free, _ = torch.cuda.mem_get_info()

    with MANIFEST.open(encoding="utf-8") as source:
        regular = {row["sample_id"]: row for line in source if (row := json.loads(line))["lighting"] == "regular"}
    require(len(regular) == 100 and len({row["files"]["normal"]["sha256"] for row in regular.values()}) == 1,
            "Expected a single audited normal reference")
    reference = regular["regular/0001"]["files"]["normal"]
    ref_rgb = load_asset(reference, "RGB")
    with WEIGHTS.open("rb") as source:
        weight_sha = hashlib.file_digest(source, "sha256").hexdigest()
    require(weight_sha.startswith("9ba9bcbe"), "TorchVision weight hash prefix mismatch")

    out = HERE / "runs" / datetime.now(timezone.utc).strftime("regular_smoke_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    print(f"Run directory {out}", flush=True)
    measurements = {}
    observed = {}
    preset = Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms()
    mean = torch.tensor(preset.mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(preset.std, device="cuda").view(1, 3, 1, 1)

    def capture_features(_module, args, output):
        observed["input"] = list(args[0].shape)
        observed["input_dtype"] = str(args[0].dtype)
        observed["features"] = {layer: list(value.shape) for layer, value in output.items()}
        observed["feature_dtypes"] = {layer: str(value.dtype) for layer, value in output.items()}

    with measure("load_backbone", measurements):
        model = make_model(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        with measure("extract_regular_normal", measurements):
            model.train()
            model.feature_extractor.eval()
            hook = model.feature_extractor.register_forward_hook(capture_features)
            for index, (y, x) in enumerate(ORIGINS):
                embedding = model(tile_tensor(ref_rgb, y, x, mean, std))
                require(embedding.shape == (4096, 1536) and embedding.dtype == torch.float32,
                        f"Unexpected embedding shape or precision {embedding.shape}, {embedding.dtype}")
                require(torch.isfinite(embedding).all().item(), "Nonfinite normal embedding")
                if index == 0:
                    hook.remove()
            rows_per_tile, dimensions = embedding.shape
            require(observed["features"] == {"layer2": [1, 512, 64, 64], "layer3": [1, 1024, 32, 32]},
                    f"Unexpected feature geometry {observed}")
            observed["embedding_per_tile"] = list(embedding.shape)
            observed["candidate_bank"] = [sum(len(x) for x in model.embedding_store), dimensions]
        print("Observed dimensions", json.dumps(observed), flush=True)
        with measure("fit_coreset", measurements):
            model.subsample_embedding(sampling_ratio=RATIO)
            model.eval()
            bank = model.memory_bank
            require(bank.shape == (int(observed["candidate_bank"][0] * RATIO), dimensions), "Unexpected coreset shape")
            require(bank.dtype == torch.float32 and torch.isfinite(bank).all().item(), "Invalid coreset values")
            observed["memory_bank"] = list(bank.shape)
            observed["memory_bank_mib"] = bank.nbytes / MIB
            query = torch.randn(QUERY_CHUNK + 1, dimensions, device=bank.device, dtype=bank.dtype)
            for k in (1, NEIGHBORS):
                actual, _ = model.nearest_neighbors(query, k)
                expected, _ = PatchcoreModel.nearest_neighbors(model, query, k)
                torch.testing.assert_close(actual, expected)

        outputs = []
        for sample in SAMPLES:
            record = regular[sample]
            rgb = load_asset(record["files"]["bubbles"], "RGB")
            mask = load_asset(record["files"]["masks"], "L")
            foreground = mask == 255
            require(foreground.sum() == record["mask_positive_pixels"] and np.all((mask == 0) | foreground),
                    f"Invalid mask {sample}")
            sample_id = sample.split("/")[-1]
            with measure(f"infer_{sample_id}", measurements):
                anomaly_map, tile_scores = infer_frame(model, rgb, mean, std)
            np.save(out / f"{sample_id}_anomaly.npy", anomaly_map, allow_pickle=False)
            outputs.append((sample_id, rgb, foreground, anomaly_map, tile_scores))

        checkpoint = out / "fitted_state.pt"
        with measure("save_reload_verify", measurements):
            torch.save(model.state_dict(), checkpoint)
            del model, bank, embedding, query
            reloaded = make_model().eval()
            reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            reloaded.memory_bank = reloaded.memory_bank.to("cuda")
            first = outputs[0]
            reloaded_map, reloaded_scores = infer_frame(reloaded, first[1], mean, std)
            np.testing.assert_allclose(reloaded_map, first[3], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(reloaded_scores, first[4], rtol=1e-6, atol=1e-6)
            observed["reload_max_abs_map_error"] = float(np.max(np.abs(reloaded_map - first[3])))
            observed["reload_max_abs_tile_score_error"] = float(np.max(np.abs(np.array(reloaded_scores) - first[4])))

    display_max = max(float(output[3].max()) for output in outputs)
    require(display_max > 0, "All saved anomaly maps are zero")
    sample_results = []
    with measure("save_visuals", measurements):
        for sample_id, rgb, foreground, anomaly_map, tile_scores in outputs:
            heat = colormaps["inferno"](np.clip(anomaly_map / display_max, 0, 1), bytes=True)[..., :3]
            Image.fromarray(heat).save(out / f"{sample_id}_heatmap.png")
            overlay = rgb.copy()
            overlay[foreground] = np.round(0.45 * overlay[foreground] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
            Image.fromarray(overlay).save(out / f"{sample_id}_ground_truth_overlay.png")
            sample_results.append({
                "sample_id": f"regular/{sample_id}",
                "anomaly_map_shape": list(anomaly_map.shape),
                "raw_map_min": float(anomaly_map.min()),
                "raw_map_max": float(anomaly_map.max()),
                "mean_inside_ground_truth": float(anomaly_map[foreground].mean()),
                "mean_outside_ground_truth": float(anomaly_map[~foreground].mean()),
                "foreground_pixels": int(foreground.sum()),
                "image_score_max_tile_score": max(tile_scores),
                "tile_scores_in_row_major_order": tile_scores,
            })

    summary = {
        "config": {
            "seed": SEED, "reference": reference, "backbone": "wide_resnet50_2.IMAGENET1K_V2",
            "backbone_weight_sha256": weight_sha, "layers": LAYERS, "precision": "FP32, TF32 disabled",
            "batch_size": 1, "rgb_mean": preset.mean, "rgb_std": preset.std, "tile": TILE,
            "overlap": OVERLAP, "stride": STRIDE, "origins_yx": ORIGINS, "resize": None, "crop": None,
            "coreset": "Anomalib sparse random projection and greedy k-center", "coreset_ratio": RATIO,
            "nearest_neighbors": NEIGHBORS, "query_chunk": QUERY_CHUNK, "bank_chunk": BANK_CHUNK,
            "feature_aggregation": "3x3 mean pooling, bilinear layer3 upsampling, channel concatenation",
            "tile_map": "nearest-neighbor upsampling and Gaussian smoothing (sigma 4 px)",
            "stitching": "average raw tile maps per pixel", "display_scale": [0, display_max], "threshold": None,
        },
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
            "anomalib": version("anomalib"), "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": props.name, "gpu_total_mib": props.total_memory / MIB,
            "gpu_free_at_start_mib": cuda_free / MIB,
            "system_ram_mib": psutil.virtual_memory().total / MIB,
        },
        "observed": observed, "measurements": measurements, "samples": sample_results,
        "capacity_estimates": capacity_estimates(rows_per_tile, dimensions),
    }
    (out / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Verified and saved", out / "report.json", flush=True)


if __name__ == "__main__":
    main()
