"""Reuse the selected V1 PatchCore candidate sampling and tile stitching."""

import smoke
import numpy as np
import torch
from anomalib.models.components import KCenterGreedy

from smoke import FRAME, ORIGINS, RATIO, TILE, load_asset, require, tile_tensor
from split_dataset import SEED

STITCHES = ("uniform", "gaussian")
SIGMA = TILE / 4
GRID = 64
DIMENSION = 1536


def sampled_indices(rng):
    """Take one or two distinct cells from each 8×8 spatial stratum."""
    strata = np.arange(64)
    first = rng.integers(64, size=64)
    second = (first + rng.integers(1, 64, size=64)) % 64
    offsets = np.stack((first, second))
    rows = 8 * (strata // 8) + offsets // 8
    cols = 8 * (strata % 8) + offsets % 8
    return (rows * GRID + cols).ravel()


def collect_candidates(model, training, mean, std, origins=ORIGINS):
    count = len(training) * len(origins)
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
        for y, x in origins:
            model(tile_tensor(image, y, x, mean, std))
            embedding = model.embedding_store.pop()
            require(embedding.shape == (GRID ** 2, DIMENSION) and embedding.dtype == torch.float32,
                    f"Unexpected tile embedding {embedding.shape}, {embedding.dtype}")
            chosen = embedding[torch.as_tensor(sampled_indices(rng), device=embedding.device)].cpu()
            primary[offset:offset + 64].copy_(chosen[:64])
            secondary[offset:offset + 64].copy_(chosen[64:])
            offset += 64
        if number % 10 == 0:
            print(f"Extracted {number}/{len(training)} {record['lighting']} training views", flush=True)
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
    require(all(torch.isfinite(block).all().item() for block in candidates.split(smoke.BANK_CHUNK)),
            "Nonfinite training candidates")
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
