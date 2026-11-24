"""Reuse V1 AnomalyDINO extraction, coreset selection and cosine scoring."""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCHCORE = HERE.parents[1] / "patchcore" / "V2"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PATCHCORE))
os.environ["HF_HOME"] = str(HERE.parent / ".hf-cache")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["MPLCONFIGDIR"] = str(HERE.parent / ".mpl-cache")

import lighting_benchmark as reference
import phase3 as pipeline
import numpy as np
import torch
import torch.nn.functional as F
from anomalib.data import InferenceBatch
from anomalib.models.image.anomaly_dino.torch_model import AnomalyDINOModel
from smoke import FRAME, MIB, ORIGINS, OVERLAP, TILE, load_asset, measure, require, tile_tensor
from split_dataset import SEED

ENCODER = "vit_small_patch14_dinov2.lvd142m"
REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
WEIGHT_HASH = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
WEIGHTS = HERE.parent / ".hf-cache" / "hub" / f"models--timm--{ENCODER}" / "snapshots" / REVISION / "model.safetensors"
PATCH = 14
PAD = 3
PADDED = TILE + 2 * PAD
GRID = PADDED // PATCH
DIM = 384
SAMPLE_COUNT = 128
RATIO = 0.10
QUERY_CHUNK = 512
BANK_CHUNK = 8192
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


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
