"""Evaluate the frozen V1 red-trained U-Net on the regular V2 cohorts."""

import shutil
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from v2_workflow import cuda_environment, require, run_cli, runtime, sha, write_json

RED_CHECKPOINT = HERE.parent / "results/V1/unet_20261124T095027_917613Z/red/best_validation.pt"
RED_CHECKPOINT_HASH = "53ceae4d67b0f9eb81f645c28d1d4655e9bd968e417204770e9288f56994a18d"


def check_environment():
    import baseline as unet

    environment = cuda_environment(unet.SEED, fp32=False)
    require(sha(RED_CHECKPOINT) == RED_CHECKPOINT_HASH, "Unexpected frozen red transfer checkpoint")
    return {**environment, "torchvision": version("torchvision")}


def build(checkpoint, train, defects):
    import numpy as np
    import torch
    import baseline as unet

    config = check_environment()
    stats = {}
    if train is not None:
        shutil.copyfile(RED_CHECKPOINT, checkpoint)
        require(sha(checkpoint) == RED_CHECKPOINT_HASH, "Transfer checkpoint copy differs")
    with unet.measured(stats, "load_transfer_checkpoint"):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(saved["epoch"] == 7, "Expected the frozen red validation checkpoint at epoch 7")
        model = unet.build_model().eval()
        model.load_state_dict(saved["model"])
    mean, std = (torch.tensor(numbers, device="cuda").view(1, 3, 1, 1)
                 for numbers in (unet.RGB_MEAN, unet.RGB_STD))
    stitcher = unet.GaussianStitcher()

    def infer(rgb):
        return stitcher.infer(model, rgb, mean, std)

    if train is not None:
        with unet.measured(stats, "smoke_inference"):
            rgb = unet.load_image(defects[0]["files"]["bubbles"], "RGB")
            score, image_score = infer(rgb)
            require(score.shape == (unet.FRAME, unet.FRAME) and score.dtype == np.float32
                    and np.isfinite(score).all() and np.isfinite(image_score), "Invalid smoke inference")
            write_json(checkpoint.with_name("smoke.json"), {"inference_id": defects[0]["sample_id"],
                       "map_dimensions": list(score.shape), "image_score": image_score, "loaded_epoch": saved["epoch"]})
    config.update({
        "protocol": "Red-to-regular supervised transfer; no V2 fitting", "backbone": "ResNet18 U-Net",
        "source_checkpoint": str(RED_CHECKPOINT.relative_to(HERE.parents[2])),
        "source_checkpoint_sha256": sha(checkpoint), "source_best_epoch": saved["epoch"], "seed": unet.SEED,
        "frame": unet.FRAME, "tile": unet.TILE, "overlap": unet.OVERLAP, "tile_origins_yx": unet.ORIGINS,
        "resize": None, "crop": None, "stitch": "gaussian", "sigma_px": unet.SIGMA,
        "rgb_mean": unet.RGB_MEAN, "rgb_std": unet.RGB_STD, "precision": "CUDA FP16 autocast; FP32 probabilities",
        "batch_size": unet.BATCH, "training_ids": [],
    })

    def measure(name, stats):
        return unet.measured(stats, name)

    return runtime(unet.load_image, infer, measure, config, stats, metric_max=1.0)


if __name__ == "__main__":
    run_cli(HERE.parent, "U-Net", build, check_environment)
