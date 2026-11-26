"""Run configurable experiments using the existing ANNAlyzer model implementations."""

import argparse
from datetime import datetime, timezone
from functools import partial
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import v2_workflow as shared
from v3_workflow import load_module, paired_validation
from workstation_config import MODELS, ROOT, custom_dataset, origins, read_json, validate, write_json
from workstation_dataset import get_records, sha


LABELS = dict(zip(MODELS, ("PatchCore", "AnomalyDINO", "U-Net", "Flownomaly"), strict=True))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patchcore/V2"))


def gpu_check(seed, threads, fp32=True):
    import torch

    info = shared.cuda_environment(seed, fp32)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    shared.require((torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda"))[0, 0].item() == 32,
                   "CUDA matrix-multiply smoke check failed.")
    return info


def geometry(config):
    frame = config["render"]["resolution"]
    training = config["training"]
    return frame, origins(frame, training["overlap"]), training["gaussian_sigma"]


def bank_modules(config):
    import smoke
    import phase3

    frame, positions, sigma = geometry(config)
    smoke.FRAME, smoke.ORIGINS = frame, positions
    phase3.ORIGINS = positions
    return smoke, phase3, phase3.Stitcher(frame, positions, sigma)


def balanced_indices(rng, grid, count):
    import numpy as np

    boundaries = np.array_split(np.arange(grid), 8)
    cells = [rng.permutation((y[:, None] * grid + x).ravel()) for y in boundaries for x in boundaries]
    result = []
    for rank in range(max(map(len, cells))):
        for index in rng.permutation(len(cells)):
            if rank < len(cells[index]):
                result.append(cells[index][rank])
                if len(result) == count:
                    return result
    raise ValueError("Tile candidate count exceeds the feature grid.")


def banks(name, light, config, checkpoint, train):
    import numpy as np
    import torch
    from anomalib.models.components import KCenterGreedy

    smoke, phase3, stitcher = bank_modules(config)
    settings = config[name]
    smoke.QUERY_CHUNK, smoke.BANK_CHUNK = settings["query_chunk"], settings["bank_chunk"]
    adapter = load_module(HERE / name / "V2/run_v2.py", f"workstation_{name}")
    stats = {}
    info = adapter.check_environment()
    if name == "patchcore":
        preset = smoke.Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms()
        preprocessor = (preset.mean, preset.std)
        make_model = lambda: smoke.make_model(torch.load(smoke.WEIGHTS, map_location="cpu", weights_only=True))
        grid, dimension = phase3.GRID, phase3.DIMENSION
    else:
        import baseline as dino

        dino.QUERY_CHUNK, dino.BANK_CHUNK = settings["query_chunk"], settings["bank_chunk"]
        preprocessor, make_model = (dino.MEAN, dino.STD), dino.make_model
        grid, dimension = dino.GRID, dino.DIM
    mean, std = (torch.tensor(values, device="cuda").view(1, 3, 1, 1) for values in preprocessor)
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    with smoke.measure("load_backbone", stats):
        model = make_model().eval()
    if train is None:
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    else:
        count = settings["candidates"]
        total = len(train) * len(smoke.ORIGINS) * count
        if int(total * settings["coreset"]) < 1:
            raise ValueError("Increase the candidate count or coreset ratio to retain at least one descriptor.")
        pool = torch.empty((total, dimension), dtype=torch.float32)
        rng = np.random.default_rng(seed)
        with torch.inference_mode(), smoke.measure("candidate_extraction", stats):
            model.train()
            (model.feature_extractor if name == "patchcore" else model.feature_encoder).eval()
            offset = 0
            for i, record in enumerate(train, 1):
                asset = record["files"]["normal"]
                shared.require(asset["label"] == "normal", "Anomalous sample in the bank fitting cohort.")
                image = smoke.load_asset(asset, "RGB")
                for y, x in smoke.ORIGINS:
                    model(smoke.tile_tensor(image, y, x, mean, std))
                    features = model.embedding_store.pop()
                    shared.require(features.shape == (grid * grid, dimension), "Unexpected feature grid.")
                    indices = balanced_indices(rng, grid, count)
                    pool[offset:offset + count].copy_(features[indices].cpu())
                    offset += count
                print(f"{LABELS[name]} {light}: extracted {i}/{len(train)} clean frames", flush=True)
        with torch.inference_mode(), smoke.measure("coreset", stats):
            candidates = pool.cuda()
            bank = (candidates if settings["coreset"] == 1 else
                    KCenterGreedy(candidates, settings["coreset"]).sample_coreset())
            if name == "anomalydino":
                bank = torch.nn.functional.normalize(bank, dim=1)
            model.memory_bank = bank
            del candidates, pool
            torch.save(model.state_dict(), checkpoint)
    model.eval()

    @torch.inference_mode()
    def infer(image):
        maps, score = phase3.infer_tiles(model, image, mean, std)
        return stitcher.apply(maps, "gaussian"), score

    info.update(protocol=f"V5 custom {light} normal-only {LABELS[name]}", precision="FP32",
                bank_dimensions=list(model.memory_bank.shape), bank_mib=model.memory_bank.nbytes / 1024**2,
                **settings)
    return shared.runtime(smoke.load_asset, infer, smoke.measure, info, stats)


def fit_unet(trainer, model, stitcher, mean, std, train, validation, settings, seed, checkpoint):
    import numpy as np
    import torch

    frames = trainer.load_frames(train)
    valid = trainer.load_frames(validation)
    targets = trainer.region_targets(valid)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], fused=True)
    scaler = torch.amp.GradScaler("cuda", enabled=settings["precision"] == "fp16")
    best, history, stats = -1.0, [], {}
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(frames) * len(trainer.ORIGINS))
        loss_total = 0.0
        with trainer.measured(stats, f"epoch_{epoch}_training"):
            for start in range(0, len(order), trainer.BATCH):
                tiles, masks = [], []
                for index in order[start:start + trainer.BATCH]:
                    frame, tile = divmod(int(index), len(trainer.ORIGINS))
                    y, x = trainer.ORIGINS[tile]
                    area = np.s_[y:y + trainer.TILE, x:x + trainer.TILE]
                    tiles.append(frames[frame][2][area])
                    masks.append(frames[frame][3][area])
                batch = trainer.tensor_rgb(tiles, mean, std)
                target = torch.from_numpy(np.stack(masks)).unsqueeze(1).cuda().float()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=trainer.AMP, enabled=trainer.AMP != torch.float32):
                    logits = model(batch)
                loss = sum(trainer.bce_dice(logits, target))
                shared.require(torch.isfinite(loss).item(), "Nonfinite training loss.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                loss_total += loss.item() * len(tiles)
                if start == 0 or start // trainer.BATCH % 100 == 0:
                    print(f"Epoch {epoch}: {start + len(tiles)}/{len(order)} tiles", flush=True)
        with trainer.measured(stats, f"epoch_{epoch}_validation"):
            result, _ = trainer.validate(model, valid, targets, stitcher, mean, std)
        ap = result["pixel_ap"]
        history.append({"epoch": epoch, "bce_plus_dice": loss_total / len(order), "validation_pixel_ap": ap})
        if ap > best:
            best = ap
            trainer.atomic_save({"model": model.state_dict(), "epoch": epoch, "pixel_ap": best}, checkpoint)
        write_json(checkpoint.with_name("training.json"), history)
        print(f"Epoch {epoch}/{settings['epochs']} · validation pixel AP {ap:.6f}", flush=True)
    return stats


def unet(light, config, checkpoint, train, validation):
    import torch

    trainer = load_module(HERE / "unet/V1/run.py", "workstation_unet")
    settings = config["unet"]
    trainer.FRAME, trainer.ORIGINS, trainer.SIGMA = geometry(config)
    trainer.BATCH = settings["batch"]
    trainer.AMP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[settings["precision"]]
    if trainer.AMP == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("The selected GPU does not support BF16 autocast.")
    if train is not None:
        from v3_workflow import RESNET_HASH

        shared.require(sha(trainer.WEIGHTS) == RESNET_HASH, "Unexpected ResNet18 pretrained weights.")
    weights = torch.load(trainer.WEIGHTS, map_location="cpu", weights_only=True) if train is not None else None
    model = trainer.build_model(weights)
    mean, std = (torch.tensor(values, device="cuda").view(1, 3, 1, 1)
                 for values in (trainer.RGB_MEAN, trainer.RGB_STD))
    stitcher = trainer.GaussianStitcher()
    stats = {} if train is None else fit_unet(trainer, model, stitcher, mean, std, train, validation, settings,
                                              config["training"]["seed"], checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()
    info = dict(protocol=f"V5 custom {light} supervised U-Net", best_epoch=saved["epoch"], **settings)
    return shared.runtime(trainer.load_image, partial(stitcher.infer, model, mean=mean, std=std),
                          lambda name, values: trainer.measured(values, name), info, stats, metric_max=1.0)


def flow(light, config, checkpoint, train, environment, previous):
    import torch
    from torch.nn import functional as F

    smoke, _, stitcher = bank_modules(config)
    runner = load_module(HERE / "flownomaly/run_v4.py", "workstation_flow")
    import model

    runner.check_weights()
    settings = config["flownomaly"]
    runner.SEED = config["training"]["seed"]
    for key, value in settings.items():
        setattr(runner, key.upper(), value)
    if train is not None:
        if len(train) < 3:
            raise ValueError("Flownomaly needs at least three training groups for fitting, selection and calibration.")
        ordered = sorted(train, key=lambda record: record["sample_id"])
        random.Random(runner.SEED).shuffle(ordered)
        extra = max(1, round(len(ordered) * 5 / 70))
        support = (ordered[:-2 * extra], ordered[-2 * extra:-extra], ordered[-extra:])
        runtime, saved = runner.build_runtime(checkpoint.parent, light, train, environment,
                                              support=support, stitcher=stitcher)
        assert saved == checkpoint
        runtime.configuration["protocol"] = f"V5 custom {light} normal-only Flownomaly"
        return runtime
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    path = checkpoint.with_name("vit_finetuned.pt")
    shared.require(sha(path) == saved["encoder_sha256"] == previous["configuration"]["encoder_state_sha256"],
                   "Flownomaly encoder differs from the saved checkpoint.")
    encoder = model.load_encoder(runner.WEIGHTS)
    encoder.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    encoder.cuda()
    velocity = model.VelocityField().cuda().eval()
    velocity.load_state_dict(saved["model"])
    center, spread = saved["center"].cuda(), saved["spread"].cuda()
    mean, std = (torch.tensor(v, device="cuda").view(1, 3, 1, 1)
                 for v in ((.485, .456, .406), (.229, .224, .225)))

    @torch.inference_mode()
    def infer(rgb):
        generator = torch.Generator(device="cuda").manual_seed(runner.SEED + 2)
        maps, scores = [], []
        for y, x in smoke.ORIGINS:
            padded = F.pad(smoke.tile_tensor(rgb, y, x, mean, std), (model.PAD,) * 4, mode="reflect")
            anomaly, score = model.tile_prediction(velocity, model.tokens(encoder, padded), center, spread,
                                                   saved["time"], generator)
            maps.append(anomaly.cpu().numpy())
            scores.append(score)
        return stitcher.apply(maps, "gaussian"), max(scores)

    return shared.runtime(smoke.load_asset, infer, smoke.measure, previous["configuration"].copy(), {})


def execute(model, action, path):
    import torch

    if action == "fit":
        config = validate(read_json(path))
        manifest_path = path.with_suffix(".split.json")
    else:
        run = path.resolve()
        config = validate(read_json(run / "job.json"))
        manifest_path = run / "split.json"
    manifest = read_json(manifest_path)
    dataset = custom_dataset(config["dataset"])
    status = read_json(dataset / "dataset.json")
    if (manifest["dataset"] != str(dataset)
            or manifest["catalog_sha256"] != sha(dataset / "samples.json")
            or manifest["configuration_sha256"] != sha(dataset / "dataset.json")
            or status["configuration"].get("workstation_render") != config["render"]):
        raise ValueError("Dataset differs from the frozen audit.")
    if manifest["near_duplicate_candidates"] and not config["split"]["accept_near_duplicates"]:
        raise ValueError(f"Review the cross-fold similarity candidates in {manifest_path} before fitting.")
    if {k: v for k, v in config["split"].items() if k != "accept_near_duplicates"} != manifest["split_settings"]:
        raise ValueError("Split settings changed after the audit.")
    lights = config["training"]["lights"]
    if not set(lights) <= set(config["render"]["lights"]) or model not in config["training"]["models"]:
        raise ValueError("Selected training lights or model do not match this job.")
    if action != "fit":
        completed = read_json(run / "complete.json")
        if completed["model"] != LABELS[model] or completed["lights"] != lights or completed["split_sha256"] != sha(manifest_path):
            raise ValueError("Select this model's completed validation run with its original split.")
        if action == "test" and any((run / light / "test").exists() for light in lights):
            raise ValueError("A test output already exists in this run.")
    env = gpu_check(config["training"]["seed"], config["training"]["cpu_threads"], model != "unet")
    shared.FRAME = config["render"]["resolution"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    if action == "fit":
        run = HERE / model / "results/V5" / f"workstation_v5_{stamp}"
        run.mkdir(parents=True)
        write_json(run / "job.json", config)
        write_json(run / "split.json", manifest)
    stage = "test" if action == "test" else "validation"
    output = run / "revalidation" / stamp if action == "validate" else run
    print(f"{LABELS[model]} {stage}: {output}", flush=True)
    for light in lights:
        target = output / light / stage
        target.mkdir(parents=True)
        checkpoint = run / light / ("flow_state.pt" if model == "flownomaly" else "fitted_state.pt")
        previous = None if action == "fit" else read_json(run / light / "validation/report.json")
        if previous is not None and (not isinstance(previous["threshold"], (int, float))
                                     or sha(checkpoint) != previous["checkpoint"]["sha256"]):
            raise ValueError("The fitted checkpoint differs from the saved validation report.")
        train = get_records(manifest, light, "train") if action == "fit" else None
        evaluate = get_records(manifest, light, stage)
        if model == "unet":
            runtime = unet(light, config, checkpoint, train, evaluate)
        elif model == "flownomaly":
            runtime = flow(light, config, checkpoint, train, env, previous)
        else:
            runtime = banks(model, light, config, checkpoint, train)
        report = paired_validation(target, light, evaluate, evaluate, runtime, stage=stage,
                                   threshold=None if previous is None else previous["threshold"])
        runtime.configuration.update(env, seed=config["training"]["seed"], frame=shared.FRAME, tile=512,
                                     sigma_px=config["training"]["gaussian_sigma"],
                                     overlap=config["training"]["overlap"])
        report.update(model=LABELS[model], configuration=runtime.configuration,
                      dataset_sha256={"workstation_split": sha(run / "split.json")},
                      fit_measurements=runtime.measurements,
                      checkpoint={"file": checkpoint.relative_to(run).as_posix(), "sha256": sha(checkpoint)})
        write_json(target / "report.json", report)
        del runtime
        torch.cuda.empty_cache()
    if action == "fit":
        code = [Path(__file__), HERE / "workstation_dataset.py", HERE / "workstation_config.py",
                HERE / "v3_workflow.py", HERE / model / ("run_v4.py" if model == "flownomaly" else
                                                        "V1/run.py" if model == "unet" else "V2/run_v2.py")]
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip() if shutil.which("git") else None
        write_json(run / "complete.json", {"model": LABELS[model], "stage": stage, "lights": lights,
                                           "split_sha256": sha(run / "split.json"),
                                           "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in code},
                                           "git_revision": revision})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("action", choices=("check", "fit", "validate", "test"))
    parser.add_argument("path", type=Path, nargs="?")
    args = parser.parse_args()
    if args.action == "check":
        gpu_check(24117, 4)
    else:
        if args.path is None:
            parser.error("Supply a job config for fit, or a completed run directory for validation or test.")
        execute(args.model, args.action, args.path)


if __name__ == "__main__":
    main()
