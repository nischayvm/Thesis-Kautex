"""PatchCore feature extraction, chunked search, and resource measurement."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("MPLCONFIGDIR", str(HERE.parent / ".mpl-cache"))
os.environ.setdefault("TQDM_MININTERVAL", "5")

import hashlib
import io
import json
from contextlib import contextmanager
from threading import Event, Thread
from time import perf_counter

import numpy as np
import psutil
import torch
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
from PIL import Image
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

DATASET = HERE.parents[2] / "data" / "annalyzer_batch_2048"
WEIGHTS = HERE.parent / "V1" / "wide_resnet50_2-9ba9bcbe.pth"
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
