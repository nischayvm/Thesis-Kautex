"""Run the isolated Flownomaly adaptation on the frozen ANNAlyzer cohorts."""

import argparse
import copy
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANOMALY = HERE.parent
sys.path[:0] = [str(ANOMALY), str(ANOMALY / "patchcore/V2")]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mpl-cache"))

import v2_workflow as shared
from v3_workflow import COLOURS, cohorts, save_validation


SEED = 24117
WEIGHT_REVISION = "4610ca143709d58a633b6397a74412c2c3842454"
WEIGHT_HASH = "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081"
WEIGHTS = (ANOMALY / "anomalydino/.hf-cache/hub/models--timm--vit_small_patch14_dinov2.lvd142m"
           / "snapshots" / WEIGHT_REVISION / "model.safetensors")
ENCODER_EPOCHS = 3
ENCODER_LR = 3e-5
FLOW_EPOCHS = 12
FLOW_LR = 1e-3
FLOW_BATCH = 512
TOKENS_PER_TILE = 64


def check_weights():
    shared.require(WEIGHTS.is_file() and shared.sha(WEIGHTS) == WEIGHT_HASH,
                   f"Expected the audited local DINOv2 weights at {WEIGHTS}")


def split_support(records):
    """Use independent frame IDs for fitting, checkpoint selection and operating-time calibration."""
    ordered = sorted(records, key=lambda record: record["sample_id"])
    ids = {record["sample_id"] for record in ordered}
    shared.require(len(ids) == len(ordered) == 70
                   and all(record["files"]["normal"]["label"] == "normal" for record in ordered),
                   "Flownomaly requires the 70 unique normal training records")
    random.Random(SEED).shuffle(ordered)
    return ordered[:60], ordered[60:65], ordered[65:]


def fine_tune_encoder(encoder, teacher, records, mean, std):
    import torch
    from torch.nn import functional as F
    import model
    from smoke import ORIGINS, load_asset, tile_tensor

    model.enable_encoder_training(encoder)
    optim = torch.optim.AdamW([p for p in encoder.parameters() if p.requires_grad], lr=ENCODER_LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    generator = torch.Generator().manual_seed(SEED)
    history = []
    for epoch in range(ENCODER_EPOCHS):
        losses = []
        encoder.train()
        for record in records:
            rgb = load_asset(record["files"]["normal"], "RGB")
            for index in torch.randperm(len(ORIGINS), generator=generator)[:5].tolist():
                tile = tile_tensor(rgb, *ORIGINS[index], mean, std)
                padded = F.pad(tile, (model.PAD,) * 4, mode="reflect")
                mask = torch.randperm(model.GRID ** 2, generator=generator)[:model.GRID ** 2 // 10].cuda()
                optim.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model.masked_token_loss(encoder, teacher, padded, mask)
                shared.require(torch.isfinite(loss).item(), "Nonfinite encoder distillation loss")
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
                losses.append(loss.item())
        average = sum(losses) / len(losses)
        history.append({"epoch": epoch + 1, "loss": average})
        print(f"ViT fine-tuning epoch {epoch + 1}/{ENCODER_EPOCHS} · masked token loss {average:.6f}", flush=True)
    encoder.requires_grad_(False).eval()
    return history


def collect_normal_views(encoder, records, mean, std, tile_limit=None):
    import torch
    from torch.nn import functional as F
    import model
    from smoke import ORIGINS, load_asset, tile_tensor

    generator = torch.Generator().manual_seed(SEED)
    views = []
    for number, record in enumerate(records, start=1):
        rgb = load_asset(record["files"]["normal"], "RGB")
        origin_indices = range(len(ORIGINS))
        if tile_limit is not None:
            origin_indices = torch.randperm(len(ORIGINS), generator=generator)[:tile_limit].tolist()
        selected = []
        for index in origin_indices:
            padded = F.pad(tile_tensor(rgb, *ORIGINS[index], mean, std), (model.PAD,) * 4, mode="reflect")
            with torch.no_grad():
                values = model.tokens(encoder, padded)[0]
            sample = torch.randperm(len(values), generator=generator)[:TOKENS_PER_TILE].cuda()
            selected.append(values[sample].cpu())
        views.append(torch.cat(selected))
        if number % 10 == 0 or number == len(records):
            print(f"Extracted Flownomaly tokens from {number}/{len(records)} normal views", flush=True)
    return views


def fit_field(field, train, validation):
    import torch
    import model

    optimizer = torch.optim.AdamW(field.parameters(), lr=FLOW_LR, weight_decay=1e-4)
    order = torch.Generator().manual_seed(SEED)
    noise = torch.Generator(device="cuda").manual_seed(SEED)
    history = []
    best, best_loss = None, float("inf")
    for epoch in range(FLOW_EPOCHS):
        field.train()
        for index in torch.randperm(len(train), generator=order).split(FLOW_BATCH):
            optimizer.zero_grad(set_to_none=True)
            loss = model.field_loss(field, train[index].cuda(), noise)
            shared.require(torch.isfinite(loss).item(), "Nonfinite flow-matching loss")
            loss.backward()
            optimizer.step()
        field.eval()
        with torch.no_grad():
            evaluation_noise = torch.Generator(device="cuda").manual_seed(SEED + 1)
            total = sum(len(block) * model.field_loss(field, block.cuda(), evaluation_noise).item()
                        for block in validation.split(FLOW_BATCH))
        score = total / len(validation)
        shared.require(torch.isfinite(torch.tensor(score)).item(), "Nonfinite normal checkpoint selection loss")
        history.append({"epoch": epoch + 1, "normal_selection_loss": score})
        if score < best_loss:
            best = {key: value.cpu().clone() for key, value in field.state_dict().items()}
            best_loss = score
        print(f"Flow fit epoch {epoch + 1}/{FLOW_EPOCHS} · normal selection loss {score:.6f}", flush=True)
    field.load_state_dict(best)
    return history


def save_checkpoint(torch, state, path):
    pending = path.with_suffix(".pending")
    torch.save(state, pending)
    pending.replace(path)


def build_runtime(folder, colour, records, environment, *, support=None, stitcher=None):
    import torch
    from torch.nn import functional as F
    import model
    from phase3 import Stitcher
    from smoke import ORIGINS, load_asset, measure, tile_tensor

    normalization = ((.485, .456, .406), (.229, .224, .225))
    mean, std = (torch.tensor(v, device="cuda").view(1, 3, 1, 1) for v in normalization)
    fit, selection, calibration = split_support(records) if support is None else support
    stats = {}
    encoder_file, field_file = folder / "vit_finetuned.pt", folder / "flow_state.pt"
    with measure("load_pretrained_vit", stats):
        encoder = model.load_encoder(WEIGHTS).cuda()
        teacher = copy.deepcopy(encoder)
    with measure("vit_fine_tuning", stats):
        encoder_history = fine_tune_encoder(encoder, teacher, fit, mean, std)
        del teacher
        save_checkpoint(torch, encoder.state_dict(), encoder_file)
        del encoder
        encoder = model.load_encoder(WEIGHTS)
        encoder.load_state_dict(torch.load(encoder_file, map_location="cpu", weights_only=True))
        encoder.cuda()
    weight_hash = shared.sha(encoder_file)
    with measure("normal_token_extraction", stats):
        train = torch.cat(collect_normal_views(encoder, fit, mean, std))
        selection_features = torch.cat(collect_normal_views(encoder, selection, mean, std))
        calibration_features = collect_normal_views(encoder, calibration, mean, std, tile_limit=5)
        center, spread = train.mean(dim=0), train.std(dim=0, correction=0).clamp_min(1e-5)
        train.sub_(center).div_(spread)
        selection_features.sub_(center).div_(spread)
    field = model.VelocityField().cuda()
    with measure("flow_matching", stats):
        field_history = fit_field(field, train, selection_features)
    del train, selection_features
    with measure("operating_time_calibration", stats):
        field.requires_grad_(False).eval()
        calibration_features = [((view - center) / spread).cuda() for view in calibration_features]
        time, sweep = model.operating_time(field, calibration_features, SEED)
        save_checkpoint(torch, {"model": field.state_dict(), "center": center, "spread": spread,
                                "time": time, "encoder_sha256": weight_hash}, field_file)
        saved = torch.load(field_file, map_location="cpu", weights_only=True)
        shared.require(saved["encoder_sha256"] == weight_hash, "Flow checkpoint refers to a different ViT")
        field.load_state_dict(saved["model"])
    shared.write_json(folder / "training.json", {"encoder": encoder_history, "field": field_history,
                                                  "operating_time": time, "calibration": sweep})
    center, spread = saved["center"].cuda(), saved["spread"].cuda()
    del calibration_features
    stitcher = Stitcher() if stitcher is None else stitcher

    @torch.inference_mode()
    def infer(rgb):
        generator = torch.Generator(device="cuda").manual_seed(SEED + 2)
        maps, scores = [], []
        for y, x in ORIGINS:
            padded = F.pad(tile_tensor(rgb, y, x, mean, std), (model.PAD,) * 4, mode="reflect")
            anomaly, score = model.tile_prediction(field, model.tokens(encoder, padded), center, spread, time, generator)
            maps.append(anomaly.cpu().numpy())
            scores.append(score)
        return stitcher.apply(maps, "gaussian"), max(scores)

    configuration = {**environment, "protocol": f"V4 {colour} Flownomaly adaptation with fine-tuned DINOv2",
                     "backbone": model.ENCODER, "initial_weight_sha256": WEIGHT_HASH, "weight_revision": WEIGHT_REVISION,
                     "encoder_state": encoder_file.name, "encoder_state_sha256": weight_hash,
                     "trainable_encoder": "Last two ViT blocks and output norm; frozen during flow fitting",
                     "feature_layers_zero_indexed": model.LAYERS, "feature_dimension": model.DIMENSION,
                     "encoder_epochs": ENCODER_EPOCHS, "encoder_learning_rate": ENCODER_LR, "encoder_tiles_per_frame_epoch": 5,
                     "flow_epochs": FLOW_EPOCHS, "flow_learning_rate": FLOW_LR, "flow_batch": FLOW_BATCH,
                     "tokens_per_fitting_tile": TOKENS_PER_TILE, "frame": shared.FRAME, "tile": model.TILE,
                     "overlap": 128, "gaussian_sigma_px": 128, "reflection_padding_px": model.PAD,
                     "seed": SEED, "operating_time": time,
                     "normal_fit_ids": [r["sample_id"] for r in fit],
                     "normal_selection_ids": [r["sample_id"] for r in selection],
                     "normal_calibration_ids": [r["sample_id"] for r in calibration]}
    return shared.runtime(load_asset, infer, measure, configuration, stats), field_file


def smoke():
    """Use CPU-only synthetic input to check ViT gradients and the recovery path."""
    from tempfile import TemporaryDirectory

    import torch
    import model

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    student = model.load_encoder(WEIGHTS)
    teacher = copy.deepcopy(student)
    model.enable_encoder_training(student)
    padded = torch.randn(1, 3, model.PADDED, model.PADDED)
    optimizer = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=ENCODER_LR)
    loss = model.masked_token_loss(student, teacher, padded, torch.arange(8))
    loss.backward()
    shared.require(any(p.grad is not None and torch.count_nonzero(p.grad).item() for p in student.blocks[-1].parameters())
                   and all(p.grad is None for p in teacher.parameters()), "Backbone fine-tuning gradients are missing")
    optimizer.step()
    shared.require(not torch.equal(student.blocks[-1].attn.qkv.weight, teacher.blocks[-1].attn.qkv.weight),
                   "The ViT optimizer step did not update the encoder")
    del optimizer, teacher
    with torch.no_grad():
        features = model.tokens(student.eval(), padded)
        shared.require(features.shape == (1, model.GRID ** 2, model.DIMENSION), "Unexpected ViT feature shape")
    field = model.VelocityField(width=32)
    field_optimizer = torch.optim.AdamW(field.parameters(), lr=FLOW_LR)
    model.field_loss(field, features[0, :12], torch.Generator().manual_seed(SEED)).backward()
    field_optimizer.step()
    center, spread = torch.zeros(model.DIMENSION), torch.ones(model.DIMENSION)
    time, _ = model.operating_time(field.eval(), [features[0, :12]], SEED, candidates=(0.25, 0.5))
    with TemporaryDirectory(prefix="flownomaly_smoke_") as temporary, torch.no_grad():
        encoder_path, field_path = Path(temporary) / "vit_finetuned.pt", Path(temporary) / "flow_state.pt"
        save_checkpoint(torch, student.state_dict(), encoder_path)
        save_checkpoint(torch, {"model": field.state_dict(), "encoder_sha256": shared.sha(encoder_path)}, field_path)
        student.load_state_dict(torch.load(encoder_path, weights_only=True))
        restored_tokens = model.tokens(student, padded)
        restored_flow = model.VelocityField(width=32).eval()
        saved = torch.load(field_path, weights_only=True)
        shared.require(saved["encoder_sha256"] == shared.sha(encoder_path), "Mismatched smoke checkpoints")
        restored_flow.load_state_dict(saved["model"])
        original_map, score = model.tile_prediction(field, features, center, spread, time,
                                                    torch.Generator().manual_seed(SEED))
        replay_map, replay_score = model.tile_prediction(restored_flow, restored_tokens, center, spread, time,
                                                         torch.Generator().manual_seed(SEED))
        torch.testing.assert_close(original_map, replay_map, rtol=0, atol=0)
        shared.require(score == replay_score and original_map.shape == (512, 512)
                       and torch.isfinite(original_map).all().item() and torch.isfinite(torch.tensor(score)).item(),
                       "Invalid smoke prediction after reload")
    print("CPU smoke passed: ViT and flow updates, JGN calibration, native tile scores and separate checkpoint reload.",
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", default="validate", choices=("validate", "check", "check-env", "smoke"))
    parser.add_argument("--lighting", nargs="+", default=["all"], choices=("all", *COLOURS))
    args = parser.parse_args()
    if "all" in args.lighting and len(args.lighting) > 1:
        parser.error("Use all by itself or list individual lighting conditions.")
    check_weights()
    if args.action == "smoke":
        smoke()
        return
    environment = shared.cuda_environment(SEED, fp32=True)
    if args.action == "check-env":
        return
    dataset, hashes = cohorts()
    lights = COLOURS if "all" in args.lighting else tuple(light for light in COLOURS if light in args.lighting)
    for colour in lights:
        split_support(dataset[colour]["train"])
    print("Verified frozen V4 input cohorts and training-only cut points.", flush=True)
    if args.action == "check":
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = HERE / "results/V4" / f"flownomaly_v4_{stamp}"
    output.mkdir(parents=True)
    shutil.copyfile(shared.DATA / "manifest_v2.json", output / "manifest_v2.json")
    shutil.copyfile(shared.SPLIT, output / "split_v1.json")
    print(f"Run directory: {output}", flush=True)
    for colour in lights:
        light = output / colour
        light.mkdir()
        print(f"Flownomaly · {colour}", flush=True)
        runtime, checkpoint = build_runtime(light, colour, dataset[colour]["train"], environment)
        save_validation(output, colour, dataset[colour], runtime, checkpoint, "Flownomaly", hashes)
        del runtime
    shared.write_json(output / "complete.json", {"model": "Flownomaly", "stage": "validation", "lights": lights,
                                                 "dataset_sha256": hashes, "source_sha256": {
                                                     file: shared.sha(HERE / file) for file in ("model.py", "run_v4.py", "uv.lock")
                                                 }})
    print(f"Completed V4 Flownomaly validation: {output}", flush=True)


if __name__ == "__main__":
    main()
