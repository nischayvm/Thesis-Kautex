"""Fit and score regular V2 views with the frozen V1 AnomalyDINO pipeline."""

import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from v2_workflow import cuda_environment, require, run_cli, runtime, sha, write_json


def check_environment():
    import baseline as model_code

    environment = cuda_environment(model_code.SEED, fp32=True)
    require(sha(model_code.WEIGHTS) == model_code.WEIGHT_HASH, "Unexpected DINOv2 weights")
    return {**environment, "anomalib": version("anomalib"), "timm": version("timm")}


def build(checkpoint, train, defects):
    import torch
    import baseline as dino

    config = check_environment()
    mean, std = (torch.tensor(numbers, device="cuda").view(1, 3, 1, 1) for numbers in (dino.MEAN, dino.STD))
    stats = {}
    if train is not None:
        with dino.measure("smoke", stats):
            smoke = dino.smoke_test(checkpoint.parent, train[0], defects[0], mean, std)
            write_json(checkpoint.with_name("smoke.json"), smoke)
        torch.cuda.empty_cache()
    with dino.measure("load_backbone", stats):
        model = dino.make_model()
    if train is not None:
        with torch.inference_mode():
            with dino.measure("candidate_extraction", stats):
                candidates = dino.collect(model, train, mean, std, dino.ORIGINS)
            with dino.measure("coreset", stats):
                dino.fit(model, candidates)
            del candidates
            dino.reference.save_model(model, checkpoint)
    else:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    torch.cuda.empty_cache()
    stitcher = dino.pipeline.Stitcher()

    @torch.inference_mode()
    def infer(rgb):
        maps, score = dino.pipeline.infer_tiles(model, rgb, mean, std)
        return stitcher.apply(maps, "gaussian"), score

    config.update({
        "protocol": "Regular V2 normal-only AnomalyDINO", "backbone": dino.ENCODER, "revision": dino.REVISION,
        "weight_sha256": dino.WEIGHT_HASH, "seed": dino.SEED, "frame": dino.FRAME, "tile": dino.TILE,
        "overlap": dino.OVERLAP, "tile_origins_yx": dino.ORIGINS, "padded_tile": dino.PADDED,
        "padding": "3 px reflection per side; removed before stitching", "resize": None, "crop": None,
        "stitch": "gaussian", "sigma_px": dino.pipeline.SIGMA, "rgb_mean": dino.MEAN, "rgb_std": dino.STD,
        "precision": "FP32", "batch_size": 1, "foreground_masking": False, "reference_augmentation": False,
        "distance": "cosine", "nearest_neighbours": 1, "candidates_per_tile": dino.SAMPLE_COUNT,
        "coreset_ratio": dino.RATIO, "query_chunk": dino.QUERY_CHUNK, "bank_chunk": dino.BANK_CHUNK,
        "bank_dimensions": list(model.memory_bank.shape), "bank_mib": model.memory_bank.nbytes / dino.MIB,
        "training_ids": [row["sample_id"] for row in train] if train else [],
    })
    return runtime(dino.load_asset, infer, dino.measure, config, stats)


if __name__ == "__main__":
    run_cli(HERE.parent, "AnomalyDINO", build, check_environment)
