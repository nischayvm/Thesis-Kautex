"""Fit and score regular V2 views with the selected V1 PatchCore configuration."""

import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from v2_workflow import cuda_environment, require, run_cli, runtime, sha, write_json

WEIGHT_HASH = "9ba9bcbebc349d733a72eb7608143fd754e4689ac8d8ce2916ca0bdff6443950"


def check_environment():
    import smoke
    from split_dataset import SEED

    environment = cuda_environment(SEED, fp32=True)
    require(sha(smoke.WEIGHTS) == WEIGHT_HASH, "Unexpected WideResNet50-2 weights")
    return {**environment, "anomalib": version("anomalib"), "torchvision": version("torchvision")}


def build(checkpoint, train, defects):
    import numpy as np
    import torch
    import phase3 as pipeline
    import smoke
    from lighting_benchmark import save_model
    from split_dataset import SEED

    config = check_environment()
    weights = torch.load(smoke.WEIGHTS, map_location="cpu", weights_only=True)
    preset = smoke.Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms()
    mean, std = (torch.tensor(numbers, device="cuda").view(1, 3, 1, 1) for numbers in (preset.mean, preset.std))
    stats = {}
    with smoke.measure("load_backbone", stats):
        model = smoke.make_model(weights).eval()
    if train is not None:
        with torch.inference_mode(), smoke.measure("smoke", stats):
            origin = smoke.ORIGINS[len(smoke.ORIGINS) // 2]
            primary, secondary = pipeline.collect_candidates(model, [train[0]], mean, std, [origin])
            model.memory_bank = pipeline.fit_bank(primary, secondary, 128)
            rgb = smoke.load_asset(defects[0]["files"]["bubbles"], "RGB")
            tile = smoke.tile_tensor(rgb, *origin, mean, std)
            expected = model(tile)
            smoke_path = checkpoint.with_name("smoke_state.pt")
            smoke_state = save_model(model, smoke_path)
            reloaded = smoke.make_model(weights).eval()
            reloaded.load_state_dict(torch.load(smoke_path, map_location="cpu", weights_only=True))
            actual = reloaded(tile)
            torch.testing.assert_close(expected.anomaly_map, actual.anomaly_map, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(expected.pred_score, actual.pred_score, atol=1e-6, rtol=1e-6)
            require(actual.anomaly_map.shape == (1, 1, smoke.TILE, smoke.TILE)
                    and torch.isfinite(actual.anomaly_map).all().item()
                    and torch.isfinite(actual.pred_score).all().item(), "Invalid smoke inference")
            write_json(checkpoint.with_name("smoke.json"), {
                "fit_id": train[0]["sample_id"], "inference_id": defects[0]["sample_id"],
                "origin_yx": origin, "tile": list(tile.shape), "bank": smoke_state["dimensions"],
                "reload_max_error": (expected.anomaly_map - actual.anomaly_map).abs().max().item(),
            })
            del reloaded, primary, secondary
        torch.cuda.empty_cache()
        with torch.inference_mode():
            with smoke.measure("candidate_extraction", stats):
                primary, secondary = pipeline.collect_candidates(model, train, mean, std)
            with smoke.measure("coreset", stats):
                model.memory_bank = pipeline.fit_bank(primary, secondary, 128)
            del primary, secondary
            save_model(model, checkpoint)
    else:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    torch.cuda.empty_cache()
    stitcher = pipeline.Stitcher()

    @torch.inference_mode()
    def infer(rgb):
        maps, score = pipeline.infer_tiles(model, rgb, mean, std)
        return stitcher.apply(maps, "gaussian"), score

    config.update({
        "protocol": "Regular V2 normal-only PatchCore", "backbone": "wide_resnet50_2.IMAGENET1K_V2",
        "weight_sha256": WEIGHT_HASH, "seed": SEED, "layers": list(smoke.LAYERS),
        "frame": smoke.FRAME, "tile": smoke.TILE, "overlap": smoke.OVERLAP, "tile_origins_yx": smoke.ORIGINS,
        "resize": None, "crop": None, "stitch": "gaussian", "sigma_px": pipeline.SIGMA,
        "rgb_mean": preset.mean, "rgb_std": preset.std, "precision": "FP32", "batch_size": 1,
        "candidates_per_tile": 128, "coreset_ratio": smoke.RATIO, "nearest_neighbours": smoke.NEIGHBORS,
        "query_chunk": smoke.QUERY_CHUNK, "bank_chunk": smoke.BANK_CHUNK,
        "bank_dimensions": list(model.memory_bank.shape), "bank_mib": model.memory_bank.nbytes / smoke.MIB,
        "training_ids": [row["sample_id"] for row in train] if train else [],
    })
    return runtime(smoke.load_asset, infer, smoke.measure, config, stats)


if __name__ == "__main__":
    run_cli(HERE.parent, "PatchCore", build, check_environment)
