"""Run all lighting conditions using the established V1/V2 model implementations."""

import argparse
import csv
import importlib.util
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import v2_workflow as shared


COLOURS = ("red", "green", "regular")
MODELS = {"patchcore": "PatchCore", "anomalydino": "AnomalyDINO", "unet": "U-Net"}
RESNET_HASH = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def load_module(path, name):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohorts():
    regular, split, audit, _, _, hashes = shared.preflight(False, True)
    legacy = shared.defect_records(split, audit, "validation")
    colour_ids = split["coloured_setup_ids"]
    shared.require({fold: len(ids) for fold, ids in colour_ids.items()} == shared.COUNTS
                   and len(set().union(*colour_ids.values())) == 100, "Invalid frozen coloured split")
    dataset = {"regular": {"train": regular["train"], "normal": regular["validation"], "defects": legacy}}
    for colour in COLOURS[:2]:
        groups = {}
        for fold in ("train", "validation"):
            records = []
            for sample in colour_ids[fold]:
                row = audit[f"{colour}/{sample}"]
                assets = row["files"]
                shared.require(row["capture_setup_id"] == f"colored/{sample}"
                               and assets["normal"]["label"] == "normal"
                               and assets["bubbles"]["label"] == "anomalous"
                               and assets["bubbles"]["mask_path"] == assets["masks"]["path"],
                               f"Invalid audited pairing for {row['sample_id']}")
                files = {}
                for kind, asset in assets.items():
                    path = shared.V1_DATA / asset["path"]
                    shared.require(shared.sha(path) == asset["sha256"], f"Asset differs from the audit: {path}")
                    files[kind] = {**asset, "path": str(path)}
                records.append({**row, "files": files})
            groups[fold] = records
        dataset[colour] = {"train": groups["train"], "normal": groups["validation"], "defects": groups["validation"]}
    return dataset, hashes


def paired_validation(folder, colour, normal, defects, runtime, *, stage="validation", threshold=None):
    """Score both classes from one lighting setup and pool their full-frame pixels."""
    import numpy as np
    import matplotlib
    from matplotlib import colormaps
    from phase3_metrics import PixelMetrics, RegionTargets
    from sklearn.metrics import average_precision_score, roc_auc_score

    raw = folder / "raw"
    visuals = folder / "visuals"
    raw.mkdir()
    visuals.mkdir()
    measurements, rows = {}, []
    inputs = [(True, record) for record in normal] + [(False, record) for record in defects]
    with runtime.measure("inference", measurements):
        for index, (clean, record) in enumerate(inputs, 1):
            stem = record["sample_id"].replace("/", "_") + ("_normal" if clean else "")
            rgb = runtime.load_rgb(record["files"]["normal" if clean else "bubbles"])
            start = perf_counter()
            score, image_score = runtime.infer(rgb)
            seconds = perf_counter() - start
            shared.require(score.shape == (shared.FRAME,) * 2 and score.dtype == np.float32
                           and np.isfinite(score).all() and score.min() >= 0 and np.isfinite(image_score),
                           f"Invalid map or score: {stem}")
            score_file = raw / f"{stem}.npy"
            np.save(score_file, score, allow_pickle=False)
            rows.append({"id": record["sample_id"], "cohort": "matched_normal" if clean else "matched_defect",
                         "image_score": float(image_score), "map_max": float(score.max()),
                         "inference_seconds": seconds, "score_map": score_file.relative_to(folder).as_posix(),
                         "visual_stem": stem})
            print(f"{colour} {stage} {index}/{len(inputs)} · {stem} · score {image_score:.6f}", flush=True)
    if threshold is None:
        shared.require(stage == "validation", "Test requires the saved validation threshold")
        threshold = max(row["image_score"] for row in rows[:len(normal)])
    display_max = max(max(row["map_max"] for row in rows), 1e-12)
    metrics = PixelMetrics(runtime.metric_max or display_max)
    empty = RegionTargets(np.zeros((shared.FRAME, shared.FRAME), dtype=bool))
    for row, (clean, record) in zip(rows, inputs, strict=True):
        target = empty
        if not clean:
            mask = runtime.load_mask(record["files"]["masks"])
            shared.require(np.isin(mask, (0, 255)).all()
                           and int(np.count_nonzero(mask)) == record["mask_positive_pixels"],
                           f"Mask differs from the audit: {record['sample_id']}")
            target = RegionTargets(mask > 0)
        score = np.load(folder / row["score_map"], allow_pickle=False)
        metrics.update(score, target)
        row["flagged"] = row["image_score"] > threshold
        rgb = runtime.load_rgb(record["files"]["normal" if clean else "bubbles"])
        shared.render_artifacts(visuals / row["visual_stem"], rgb, target.foreground, score, display_max, colormaps["magma"])
    negatives, positives = rows[:len(normal)], rows[len(normal):]
    fp = sum(row["flagged"] for row in negatives)
    tp = sum(row["flagged"] for row in positives)
    labels = [int(not clean) for clean, _ in inputs]
    scores = [row["image_score"] for row in rows]
    with (folder / "scores.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    for position, collection, label in ((0, negatives, "Normals"), (1, positives, "Defects")):
        ax.scatter([row["image_score"] for row in collection], [position] * len(collection), label=label)
    ax.axvline(threshold, linestyle="--", color="black", label="Validation threshold")
    ax.set(yticks=[], xlabel="Raw image score", title=f"{colour.title()} lighting · {stage}")
    ax.legend()
    fig.savefig(folder / "image_scores.png", dpi=150)
    plt.close(fig)
    return {"stage": stage, "lighting": colour, "pixel_scope": "all",
            "image_auroc": roc_auc_score(labels, scores), "image_ap": average_precision_score(labels, scores),
            "threshold": threshold, "threshold_rule": "Flag score > maximum normal score on same-light validation",
            "display_max": display_max, "normal_cohort": {"frames": len(normal), "false_positives": fp,
                                                          "false_positive_rate": fp / len(normal)},
            "defect_cohort": {"frames": len(defects), "true_positives": tp, "sensitivity": tp / len(defects),
                              **metrics.compute()}, "measurements": measurements, "rows": rows}


def unet_setup(folder):
    # Load the archived trainer without invoking its V1-specific command-line entry point.
    os.environ["MPLCONFIGDIR"] = str(folder / ".mpl-cache")
    import matplotlib

    matplotlib.get_cachedir()
    sys.path.insert(0, str(shared.ANOMALY / "patchcore/V2"))
    trainer = load_module(folder / "V1/run.py", "unet_v3_trainer")
    shared.require(shared.sha(trainer.WEIGHTS) == RESNET_HASH, "Unexpected ResNet18 pretrained weights")
    config = shared.cuda_environment(trainer.SEED, fp32=False)
    trainer.torch.set_num_threads(4)
    trainer.torch.backends.cudnn.benchmark = False
    return trainer, config


def validate_unet(folder, output, dataset, hashes, trainer, base_config):
    torch = trainer.torch
    mean, std = (torch.tensor(values, device="cuda").view(1, 3, 1, 1)
                 for values in (trainer.RGB_MEAN, trainer.RGB_STD))
    stitcher = trainer.GaussianStitcher()
    state = torch.load(trainer.WEIGHTS, map_location="cpu", weights_only=True)
    training = output / "training"
    training.mkdir()
    smoke = trainer.smoke_test(state, dataset["red"]["train"][0], mean, std)
    shared.write_json(output / "smoke.json", smoke)
    identity = {**hashes, "weights": RESNET_HASH, "trainer": shared.sha(folder / "V1/run.py"),
                "architecture": shared.sha(folder / "V1/model.py"), "workflow": shared.sha(Path(__file__))}
    red = None
    for colour in COLOURS:
        cohort = dataset[colour]
        fit = (trainer.fit_colour(training, colour, cohort["train"], cohort["defects"], state, stitcher, mean,
                                  std, identity) if colour != "regular" else red)
        if colour == "red":
            red = fit
        checkpoint = training / fit["best_checkpoint"]
        model = trainer.build_model().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])

        def infer(rgb):
            return stitcher.infer(model, rgb, mean, std)

        config = {**base_config, "protocol": ("V3 red-to-regular transfer; no regular defect training"
                                              if colour == "regular" else f"V3 supervised within-{colour}"),
                  "best_epoch": fit["best_epoch"], "epochs": trainer.EPOCHS, "seed": trainer.SEED,
                  "learning_rate": trainer.LR, "frame": trainer.FRAME, "tile": trainer.TILE,
                  "overlap": trainer.OVERLAP, "sigma_px": trainer.SIGMA, "batch_size": trainer.BATCH}
        measurements = {} if colour == "regular" else dict(fit["measurements"])
        if colour != "regular":
            for epoch in fit["history"]:
                measurements.update({f"epoch_{epoch['epoch']}_{key}": row
                                     for key, row in epoch["measurements"].items()})
        runtime = shared.runtime(trainer.load_image, infer, lambda name, stats: trainer.measured(stats, name),
                                 config, measurements, metric_max=1.0)
        save_validation(output, colour, cohort, runtime, checkpoint, MODELS["unet"], hashes)
        del runtime, model
        torch.cuda.empty_cache()


def save_validation(output, colour, cohort, runtime, checkpoint, label, hashes):
    folder = output / colour / "validation"
    folder.mkdir(parents=True, exist_ok=False)
    if colour == "regular":
        result = shared.evaluate("validation", folder, cohort["normal"], cohort["defects"], runtime)
        result.update({"lighting": colour, "pixel_scope": "defects",
                       "threshold_rule": "Flag score > maximum regular validation normal score"})
    else:
        result = paired_validation(folder, colour, cohort["normal"], cohort["defects"], runtime)
    result.update({"model": label, "configuration": runtime.configuration, "fit_measurements": runtime.measurements,
                   "dataset_sha256": hashes, "checkpoint": {"file": checkpoint.relative_to(output).as_posix(),
                                                            "sha256": shared.sha(checkpoint)}})
    shared.write_json(folder / "report.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("action", nargs="?", default="validate", choices=("validate", "check", "check-env"))
    args = parser.parse_args()
    folder = shared.ANOMALY / args.model
    if args.action != "check-env":
        dataset, hashes = cohorts()
        print("Verified regular, red and green validation cohorts and frozen training assignments.", flush=True)
        if args.action == "check":
            return
    if args.model == "unet":
        trainer, config = unet_setup(folder)
    else:
        runner = load_module(folder / "V2/run_v2.py", f"{args.model}_v3_adapter")
        if args.action == "check-env":
            runner.check_environment()
    if args.action == "check-env":
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = folder / "results" / "V3" / f"all_lights_v3_{stamp}"
    output.mkdir(parents=True)
    shutil.copyfile(shared.DATA / "manifest_v2.json", output / "manifest_v2.json")
    shutil.copyfile(shared.SPLIT, output / "split_v1.json")
    print(f"Run directory: {output}", flush=True)
    if args.model == "unet":
        validate_unet(folder, output, dataset, hashes, trainer, config)
    else:
        for colour in COLOURS:
            group = dataset[colour]
            light = output / colour
            light.mkdir()
            checkpoint = light / "fitted_state.pt"
            print(f"Fitting {MODELS[args.model]} · {colour}", flush=True)
            runtime = runner.build(checkpoint, group["train"], group["defects"])
            runtime.configuration["protocol"] = f"V3 {colour} normal-only {MODELS[args.model]}"
            save_validation(output, colour, group, runtime, checkpoint, MODELS[args.model], hashes)
            del runtime
    sources = shared.source_hashes(folder)
    workflow = Path(__file__).resolve()
    sources[workflow.relative_to(shared.ROOT).as_posix()] = shared.sha(workflow)
    if args.model == "unet":
        for source in (folder / "V1/run.py", folder / "V1/model.py"):
            sources[source.relative_to(shared.ROOT).as_posix()] = shared.sha(source)
    shared.write_json(output / "complete.json", {"model": MODELS[args.model], "stage": "validation",
                                                 "lights": COLOURS, "dataset_sha256": hashes,
                                                 "source_sha256": sources})
    print(f"Completed all V3 lighting conditions: {output}", flush=True)


if __name__ == "__main__":
    main()
