"""Train two supervised U-Nets and evaluate the frozen validation cohorts."""

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

HERE = Path(__file__).resolve().parent
PATCHCORE = HERE.parent / "patchcore"
ANOMALYDINO = HERE.parent / "anomalydino"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PATCHCORE))
os.environ["MPLCONFIGDIR"] = str(HERE / ".mpl-cache")
os.environ["HF_HUB_OFFLINE"] = "1"

import cv2
import matplotlib
import numpy as np
import psutil
import torch
from matplotlib import colormaps
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score

from model import DECODER_CHANNELS, ResNet18UNet, bce_dice
from phase3_metrics import BINS, DIAGNOSTIC_FPR, FPR_LIMIT, SIZE_LABELS, PixelMetrics, RegionTargets
from split_dataset import SEED, SPLIT, load_audit, split_text

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DATA = HERE.parents[1] / "data" / "annalyzer_batch_2048"
WEIGHTS = HERE / "resnet18-f37072fd.pth"
PC_REPORT = PATCHCORE / "runs/lighting_benchmark_20261124T084630_908213Z/report.json"
DINO_REPORT = ANOMALYDINO / "runs/anomalydino_20261124T091730_300718Z/report.json"
FRAME = 2048
TILE = 512
OVERLAP = 128
STRIDE = TILE - OVERLAP
SIGMA = 128
ORIGINS = [(y, x) for y in range(0, FRAME - TILE + 1, STRIDE) for x in range(0, FRAME - TILE + 1, STRIDE)]
BATCH = 4
EPOCHS = 10
LR = 1e-4
AMP = torch.float16
MIB = 1024 ** 2
COLOURS = ("regular", "red", "green")
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def atomic_save(value, path):
    pending = path.with_suffix(".pending")
    torch.save(value, pending)
    pending.replace(path)


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


def load_frames(records, include_normal=True):
    frames = []
    zero = np.zeros((FRAME, FRAME), dtype=bool)
    for record in records:
        files = record["files"]
        if include_normal:
            require(files["normal"]["label"] == "normal", "Invalid normal role")
            frames.append((record["sample_id"], "normal", load_image(files["normal"], "RGB"), zero))
        mask = load_image(files["masks"], "L")
        foreground = mask == 255
        require(np.all((mask == 0) | foreground) and foreground.sum() == record["mask_positive_pixels"],
                f"Invalid mask for {record['sample_id']}")
        frames.append((record["sample_id"], "bubbles", load_image(files["bubbles"], "RGB"), foreground))
    return frames


def tensor_rgb(tiles, mean, std):
    array = np.stack(tiles)
    value = torch.from_numpy(array).permute(0, 3, 1, 2).to("cuda", dtype=torch.float32)
    return value.div_(255).sub_(mean).div_(std)


def train_epoch(model, frames, optimizer, scaler, mean, std, colour, epoch):
    model.train()
    tile_count = len(ORIGINS)
    order = np.random.default_rng(SEED + epoch).permutation(len(frames) * tile_count)
    require(len(order) % BATCH == 0, "Expected full training batches")
    running = torch.zeros(2, device="cuda")
    n_batches = len(order) // BATCH
    for step, block in enumerate(order.reshape(-1, BATCH), 1):
        images, targets = [], []
        for flat in block:
            frame, tile = divmod(int(flat), tile_count)
            y, x = ORIGINS[tile]
            area = np.s_[y:y + TILE, x:x + TILE]
            images.append(frames[frame][2][area])
            targets.append(frames[frame][3][area])
        inp = tensor_rgb(images, mean, std)
        truth = torch.from_numpy(np.stack(targets)).unsqueeze(1).to("cuda", dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=AMP):
            logits = model(inp)
        bce, dice = bce_dice(logits, truth)
        scaler.scale(bce + dice).backward()
        scaler.step(optimizer)
        scaler.update()
        running += torch.stack((bce.detach(), dice.detach()))
        if step in (1, n_batches) or step % 175 == 0:
            values = (running / step).tolist()
            require(np.isfinite(values).all(), f"Nonfinite loss in {colour} epoch {epoch}")
            print(f"{colour} epoch {epoch:02d}/{EPOCHS} · {step}/{n_batches} batches · loss {sum(values):.5f}", flush=True)
    return {"bce": values[0], "dice_loss": values[1], "bce_plus_dice": sum(values), "batches": n_batches,
            "tiles": len(order), "loss_scale_end": scaler.get_scale()}


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


def validate(model, frames, targets, stitcher, mean, std, raw_output=None):
    model.eval()
    metrics = PixelMetrics(1.0)
    rows = []
    for (sample, role, rgb, _), target in zip(frames, targets, strict=True):
        start = perf_counter()
        score, image_score = stitcher.infer(model, rgb, mean, std)
        duration = perf_counter() - start
        metrics.update(score, target)
        row = {"sample_id": sample, "role": role, "image_score": image_score, "inference_s": duration}
        if raw_output is not None:
            name = f"{sample.split('/')[-1]}_{role}_probability.npy"
            np.save(raw_output / name, score, allow_pickle=False)
            row["probability_path"] = f"raw/{name}"
        rows.append(row)
    result = metrics.compute()
    labels = [row["role"] == "bubbles" for row in rows]
    scores = [row["image_score"] for row in rows]
    result["image_auroc"] = float(roc_auc_score(labels, scores)) if not all(labels) else None
    result["image_ap"] = float(average_precision_score(labels, scores)) if not all(labels) else None
    result["mean_inference_s"] = float(np.mean([row["inference_s"] for row in rows]))
    return result, rows


def region_targets(frames):
    empty = RegionTargets(np.zeros((FRAME, FRAME), dtype=bool))
    return [empty if row[1] == "normal" else RegionTargets(row[3]) for row in frames]


def build_model(pretrained=None):
    return ResNet18UNet(pretrained).to(device="cuda", memory_format=torch.channels_last)


def smoke_test(state, record, mean, std):
    torch.manual_seed(SEED)
    frames = load_frames([record])
    model = build_model(state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    images = [frame[2][:TILE, :TILE] for frame in frames for _ in range(2)]
    targets = [frame[3][:TILE, :TILE] for frame in frames for _ in range(2)]
    inp = tensor_rgb(images, mean, std)
    target = torch.from_numpy(np.stack(targets)).unsqueeze(1).to("cuda", dtype=torch.float32)
    with torch.autocast("cuda", dtype=AMP):
        logits = model(inp)
    bce, dice = bce_dice(logits, target)
    require(logits.shape == (BATCH, 1, TILE, TILE) and torch.isfinite(bce + dice).item(), "Smoke step failed")
    scaler.scale(bce + dice).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"AMP smoke test passed at batch {BATCH}, loss {(bce + dice).item():.5f}", flush=True)
    return {"logit_shape": list(logits.shape), "logit_dtype": str(logits.dtype), "bce_plus_dice": (bce + dice).item()}


def fit_colour(out, colour, train, validation, state, stitcher, mean, std, run_identity):
    folder = out / colour
    folder.mkdir(exist_ok=True)
    report = folder / "report.json"
    if report.is_file():
        return json.loads(report.read_text(encoding="utf-8"))
    stats = {}
    with measured(stats, "load_data"):
        train_frames = load_frames(train)
        valid_frames = load_frames(validation)
        targets = region_targets(valid_frames)
    torch.manual_seed(SEED)
    model = build_model(state)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
    scaler = torch.amp.GradScaler("cuda")
    latest, best = folder / "latest_training.pt", folder / "best_validation.pt"
    history = []
    best_ap, best_epoch = -1.0, None
    if latest.is_file():
        resume = torch.load(latest, map_location="cpu", weights_only=True)
        require(resume["identity"] == run_identity, "Resume configuration changed")
        model.load_state_dict(resume["model"])
        opt.load_state_dict(resume["optimizer"])
        scaler.load_state_dict(resume["scaler"])
        history = resume["history"]
        best_ap, best_epoch = resume["best_ap"], resume["best_epoch"]
        del resume
    for epoch in range(len(history) + 1, EPOCHS + 1):
        epoch_resources = {}
        with measured(epoch_resources, "training"):
            training = train_epoch(model, train_frames, opt, scaler, mean, std, colour, epoch)
        with measured(epoch_resources, "validation"):
            result, _ = validate(model, valid_frames, targets, stitcher, mean, std)
        ap = result["pixel_ap"]
        promoted = ap > best_ap
        if promoted:
            best_ap, best_epoch = ap, epoch
            atomic_save({"model": model.state_dict(), "epoch": epoch, "pixel_ap": ap, "identity": run_identity}, best)
        history.append({"epoch": epoch, **training, "validation_pixel_ap": ap, "measurements": epoch_resources})
        atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                     "history": history, "best_ap": best_ap, "best_epoch": best_epoch, "identity": run_identity}, latest)
        (folder / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        print(f"{colour} epoch {epoch:02d} complete · val pixel AP {ap:.6f} · train {epoch_resources['training']['seconds']:.1f} s"
              f" · best epoch {best_epoch}", flush=True)
    del train_frames, opt, scaler
    model.load_state_dict(torch.load(best, map_location="cpu", weights_only=True)["model"])
    torch.cuda.empty_cache()
    raw = folder / "raw"
    raw.mkdir(exist_ok=True)
    with measured(stats, "best_checkpoint_validation"):
        metrics, predictions = validate(model, valid_frames, targets, stitcher, mean, std, raw)
    require(abs(metrics["pixel_ap"] - best_ap) < 1e-6, "Best checkpoint does not reproduce its pixel AP")
    record = {"protocol": f"supervised within-{colour}", "train_ids": [r["sample_id"] for r in train],
              "training_normal_count": len(train), "training_defect_count": len(train),
              "validation_ids": [r["sample_id"] for r in validation], "best_epoch": best_epoch,
              "best_checkpoint": best.relative_to(out).as_posix(), "checkpoint_sha256": sha(best),
              "metrics": metrics, "predictions": predictions, "history": history, "measurements": stats}
    report.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def transfer_regular(out, validation, stitcher, mean, std):
    folder = out / "regular"
    (folder / "raw").mkdir(parents=True, exist_ok=True)
    stats = {}
    best = out / "red" / "best_validation.pt"
    model = build_model()
    picked = torch.load(best, map_location="cpu", weights_only=True)
    model.load_state_dict(picked["model"])
    frames = load_frames(validation, include_normal=False)
    with measured(stats, "red_to_regular_validation"):
        metrics, predictions = validate(model, frames, region_targets(frames), stitcher, mean, std, folder / "raw")
    result = {"protocol": "red-trained U-Net transferred to regular; zero regular fit frames",
              "training_normal_count": 0, "training_defect_count": 0,
              "validation_ids": [r["sample_id"] for r in validation], "source_checkpoint": best.relative_to(out).as_posix(),
              "checkpoint_sha256": sha(best), "source_epoch": picked["epoch"],
              "metrics": metrics, "predictions": predictions, "measurements": stats}
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, help="Resume a prior run from its latest epoch checkpoint")
    args = parser.parse_args()
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    require(torch.cuda.is_available() and torch.version.cuda == "12.8", "Expected CUDA 12.8")
    torch.backends.cudnn.benchmark = False
    records, audit_sha = load_audit()
    split_content = SPLIT.read_text(encoding="utf-8")
    require(split_content == split_text(records, audit_sha), "Frozen split differs from audit")
    split = json.loads(split_content)
    pc = json.loads(PC_REPORT.read_text(encoding="utf-8"))
    dino = json.loads(DINO_REPORT.read_text(encoding="utf-8"))
    split_hash = sha(SPLIT)
    require(pc["configuration"]["split_sha256"] == dino["configuration"]["split_sha256"] == split_hash,
            "Frozen baseline splits differ")
    require(sha(WEIGHTS).startswith("f37072fd"), "Pretrained ResNet18 checksum mismatch")
    dino_configuration, pc_configuration = dino["configuration"], pc["configuration"]
    require(dino_configuration["tile_origins_yx"] == [list(p) for p in ORIGINS]
            and dino_configuration["gaussian_sigma_px"] == pc_configuration["gaussian_sigma_px"] == SIGMA,
            "Existing tile coordinates or stitch differs")
    coloured = split["coloured_setup_ids"]
    require([len(coloured[k]) for k in ("train", "validation", "test")] == [70, 15, 15]
            and set(coloured["train"] + coloured["validation"]).isdisjoint(coloured["test"]), "Invalid split")
    train, val = {}, {}
    for colour in ("red", "green"):
        train[colour] = [records[f"{colour}/{key}"] for key in coloured["train"]]
        val[colour] = [records[f"{colour}/{key}"] for key in coloured["validation"]]
    val["regular"] = [records[k] for k in split["regular"]["defective_validation_ids"]]
    require(len(val["regular"]) == 3 and set(split["regular"]["defective_validation_ids"]).isdisjoint(
        split["regular"]["defective_test_ids"]), "Regular validation overlaps held-out defects")
    require(all([r["sample_id"] for r in val[c]] == pc["lighting"][c]["validation_ids"]
                == dino["lighting"][c]["validation_ids"] for c in COLOURS), "Validation cohort changed")
    out = args.resume.resolve() if args.resume else HERE / "runs" / datetime.now(timezone.utc).strftime("unet_%Y%m%dT%H%M%S_%fZ")
    require(out.parent == HERE / "runs", "Run path must be in the local U-Net runs directory")
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    if not args.resume:
        shutil.copyfile(SPLIT, out / SPLIT.name)
    print(f"Run directory {out}", flush=True)
    identity = {"split_sha256": split_hash, "weight_sha256": sha(WEIGHTS), "runner_sha256": sha(Path(__file__)),
                "model_sha256": sha(HERE / "model.py"), "lock_sha256": sha(HERE / "uv.lock")}
    configuration = {**identity, "architecture": "U-Net with ImageNet1K V1 ResNet18", "decoder_channels": DECODER_CHANNELS,
                     "trainable_encoder": True, "loss": "BCE with logits plus per-tile soft Dice loss, smooth 1",
                     "optimizer": "AdamW", "weight_decay": 0.01, "learning_rate": LR, "epochs": EPOCHS,
                     "batch_size": BATCH, "precision": "CUDA FP16 autocast with GradScaler; FP32 losses and maps",
                     "seed": SEED, "tile": TILE, "stride": STRIDE, "overlap": OVERLAP, "gaussian_sigma_px": SIGMA,
                     "rgb_mean": RGB_MEAN, "rgb_std": RGB_STD, "image_score": "max tile mean of highest 1% probabilities",
                     "checkpoint_selection": "max same-light validation pixel AP; earliest on ties",
                     "histogram_bins": BINS, "probability_range": [0, 1], "aupro_fpr_limit": FPR_LIMIT,
                     "diagnostic_fpr": DIAGNOSTIC_FPR, "regular_protocol": "red-to-regular transfer"}
    mean = torch.tensor(RGB_MEAN, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(RGB_STD, device="cuda").view(1, 3, 1, 1)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    smoke_path = out / "smoke.json"
    if not smoke_path.is_file():
        smoke_path.write_text(json.dumps(smoke_test(state, train["red"][0], mean, std), indent=2) + "\n", encoding="utf-8")
    torch.cuda.empty_cache()
    stitcher = GaussianStitcher()
    results = {}
    for colour in ("red", "green"):
        results[colour] = fit_colour(out, colour, train[colour], val[colour], state, stitcher, mean, std, identity)
    results["regular"] = transfer_regular(out, val["regular"], stitcher, mean, std)
    report = {"configuration": configuration,
              "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "torchvision": version("torchvision"), "cuda": torch.version.cuda,
                              "gpu": torch.cuda.get_device_name()},
              "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
              "baseline_report_hashes": {"patchcore": sha(PC_REPORT), "anomalydino": sha(DINO_REPORT)}, "lighting": results}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved U-Net validation report to {out}", flush=True)


if __name__ == "__main__":
    main()
