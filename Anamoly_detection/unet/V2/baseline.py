"""Reuse the V1 U-Net architecture, normalization and Gaussian inference."""

import hashlib
import io
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

HERE = Path(__file__).resolve().parent
PATCHCORE = HERE.parents[1] / "patchcore" / "V2"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PATCHCORE))
os.environ["MPLCONFIGDIR"] = str(HERE.parent / ".mpl-cache")
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import psutil
import torch
from PIL import Image

from model import ResNet18UNet
from split_dataset import SEED


DATA = HERE.parents[2] / "data" / "annalyzer_batch_2048"
FRAME = 2048
TILE = 512
OVERLAP = 128
STRIDE = TILE - OVERLAP
SIGMA = 128
ORIGINS = [(y, x) for y in range(0, FRAME - TILE + 1, STRIDE) for x in range(0, FRAME - TILE + 1, STRIDE)]
BATCH = 4
AMP = torch.float16
MIB = 1024 ** 2
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


@contextmanager
def measured(stats, name):
    process = psutil.Process()
    stop = Event()
    peak = process.memory_info().rss

    def poll():
        nonlocal peak
        while not stop.wait(0.1):
            peak = max(peak, process.memory_info().rss)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    monitor = Thread(target=poll, daemon=True)
    monitor.start()
    try:
        yield
    finally:
        stop.set()
        monitor.join()
        torch.cuda.synchronize()
        stats[name] = {"seconds": perf_counter() - start,
                       "rss_peak_mib": max(peak, process.memory_info().rss) / MIB,
                       "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
                       "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB}


def load_image(asset, mode):
    data = (DATA / asset["path"]).read_bytes()
    require(hashlib.sha256(data).hexdigest() == asset["sha256"], f"Audit hash mismatch for {asset['path']}")
    with Image.open(io.BytesIO(data)) as image:
        require(image.format == "PNG" and image.mode == mode and image.size == (FRAME, FRAME), asset["path"])
        return np.array(image)


def tensor_rgb(tiles, mean, std):
    array = np.stack(tiles)
    value = torch.from_numpy(array).permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
    return value.div_(255).sub_(mean).div_(std)


class GaussianStitcher:
    def __init__(self):
        centered = np.arange(TILE, dtype=np.float32) - (TILE - 1) / 2
        gaussian = np.exp(-0.5 * (centered / SIGMA) ** 2)
        self.weights = np.outer(gaussian, gaussian)
        self.denominator = np.zeros((FRAME, FRAME), dtype=np.float32)
        for y, x in ORIGINS:
            self.denominator[y:y + TILE, x:x + TILE] += self.weights
        require(self.denominator.min() > 0, "Stitch has uncovered pixels")

    @torch.inference_mode()
    def infer(self, model, rgb, mean, std):
        accumulator = np.zeros((FRAME, FRAME), dtype=np.float32)
        running_score = 0.0
        for offset in range(0, len(ORIGINS), BATCH):
            origins = ORIGINS[offset:offset + BATCH]
            inp = tensor_rgb([rgb[y:y + TILE, x:x + TILE] for y, x in origins], mean, std)
            with torch.autocast("cuda", dtype=AMP):
                logits = model(inp)
            require(logits.shape == (len(origins), 1, TILE, TILE), "Unexpected model map dimensions")
            probabilities = logits.float().sigmoid()[:, 0]
            top = probabilities.flatten(1).topk(TILE * TILE // 100, sorted=False).values
            running_score = max(running_score, top.mean(dim=1).amax().item())
            for (y, x), tile in zip(origins, probabilities.cpu().numpy(), strict=True):
                accumulator[y:y + TILE, x:x + TILE] += tile * self.weights
        np.divide(accumulator, self.denominator, out=accumulator)
        np.clip(accumulator, 0, 1, out=accumulator)
        return accumulator, running_score

def build_model(pretrained=None):
    return ResNet18UNet(pretrained).to(device="cuda", memory_format=torch.channels_last)
