"""Shared dataset checks and regular-light V2 validation/test protocol."""

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

ANOMALY = Path(__file__).resolve().parent
ROOT = ANOMALY.parent
DATA = ROOT / "data" / "annalyzer_regular_normals_v2"
V1_DATA = ROOT / "data" / "annalyzer_batch_2048"
SPLIT = ANOMALY / "patchcore" / "V1" / "split_v1.json"
AUDIT = ANOMALY / "dataset_audit_manifest_v1.jsonl"
COUNTS = {"train": 70, "validation": 15, "test": 15}
FRAME = 2048
# Bind the click-to-run decision to the reviewed normal manifest and duplicate audit.
REVIEWED_V2 = {
    "manifest_v2": "75f9254e20c003d5d591432c78423837d94a78e5b525abc08e292aafa6bd94b3",
    "dataset_v2": "600888c4be43b3491aee107347f2437805afc8a9920669960cf804ef0a9cae1f",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def normal_record(row):
    return {
        "sample_id": row["image_id"], "lighting": "regular",
        "files": {"normal": {**row, "path": str(ROOT / row["path"])}},
    }


def preflight(allow_near_duplicates, use_reviewed_split=False):
    for path in (DATA / "dataset.json", DATA / "manifest_v2.json"):
        require(path.is_file(), f"Missing dataset metadata: {path}")
    state = read_json(DATA / "dataset.json")
    manifest = read_json(DATA / "manifest_v2.json")
    require(state["complete"] is True,
            f"V2 render is incomplete ({len(manifest)}/100 reference or rendered records).")
    files = {"manifest_v2": DATA / "manifest_v2.json", "dataset_v2": DATA / "dataset.json",
             "split_v1": SPLIT, "audit_v1": AUDIT}
    hashes = {name: sha(path) for name, path in files.items()}
    if use_reviewed_split:
        require(all(hashes[name] == digest for name, digest in REVIEWED_V2.items()),
                "V2 metadata differs from the reviewed split. Recheck the duplicate audit before accepting it.")
    config = state["configuration"]
    sources = config["source_sha256"]
    split = read_json(SPLIT)
    require(hashes["split_v1"] == sources[SPLIT.name]
            and hashes["audit_v1"] == sources[AUDIT.name] == split["audit_manifest_sha256"],
            "Frozen V1 split or audit digest changed")
    require(config["schema"] == 2 and config["resolution"] == [FRAME, FRAME]
            and config["counts_including_reference"] == COUNTS, "Unexpected V2 dataset configuration")
    audit = {row["sample_id"]: row for row in map(json.loads, AUDIT.read_text(encoding="utf-8").splitlines())}
    omitted = config["omitted_v1_train_setup"]
    require(omitted == max(split["coloured_setup_ids"]["train"]), "Unexpected omitted training setup")
    expected = {sample: fold for fold, ids in split["coloured_setup_ids"].items()
                for sample in ids if sample != omitted}
    expected["reference"] = "train"
    require(manifest.keys() == expected.keys(), "V2 manifest does not match expected setup IDs")
    fixed = audit[split["regular"]["reference_sample_id"]]
    fixed_asset = fixed["files"]["normal"]
    groups, digests = set(), set()
    normal = {fold: [] for fold in COUNTS}
    from PIL import Image

    for sample, fold in sorted(expected.items()):
        row = manifest[sample]
        require(row["split"] == fold and row["lighting"] == "regular" and row["label"] == "normal"
                and row["mask_path"] is None and row["image_id"] == f"regular-v2/{sample}",
                f"Invalid V2 normal assignment: {sample}")
        if sample == "reference":
            path = V1_DATA / fixed_asset["path"]
            group = fixed["capture_setup_id"]
            require(row["sha256"] == fixed_asset["sha256"], "The reference differs from the V1 reference")
        else:
            path = DATA / "normal" / f"{sample}.png"
            group = audit[f"red/{sample}"]["capture_setup_id"]
            require(group == audit[f"green/{sample}"]["capture_setup_id"], f"Colour group differs: {sample}")
        require(row["group_id"] == group and group not in groups
                and row["path"] == path.relative_to(ROOT).as_posix(), f"Unexpected group or path: {sample}")
        digest = sha(path)
        require(digest == row["sha256"] and digest not in digests, f"Duplicate or changed normal: {path}")
        with Image.open(path) as image:
            require(image.format == "PNG" and image.mode == "RGB" and image.size == (FRAME, FRAME),
                    f"Invalid native normal image: {path}")
            image.verify()
        groups.add(group)
        digests.add(digest)
        normal[fold].append(normal_record(row))
    require({fold: len(rows) for fold, rows in normal.items()} == COUNTS, "Unexpected V2 split counts")
    quality = state["split_audit"]
    require(quality["exact_duplicates"] == 0, "Renderer found decoded-pixel duplicates")
    near = quality["cross_split_candidates"]
    require(not near or allow_near_duplicates or use_reviewed_split,
            f"Review {len(near)} flagged cross-split near-duplicate pairs in dataset.json. "
            "Use --accept-near-duplicates only after reviewing them.")
    return normal, split, audit, manifest, state, hashes


def defect_records(split, audit, stage):
    from PIL import Image

    ids = split["regular"][f"defective_{stage}_ids"]
    require(len(ids) == len(set(ids)) == (3 if stage == "validation" else 97),
            f"Unexpected fixed-pose {stage} defect count")
    require(not set(ids).intersection(split["regular"]["defective_test_ids" if stage == "validation"
                                                       else "defective_validation_ids"]),
            "Fixed-pose defect IDs overlap between validation and test")
    rows = []
    for sample in ids:
        row = audit[sample]
        assets = row["files"]
        require(row["capture_setup_id"] == "regular/fixed" and row["lighting"] == "regular"
                and assets["bubbles"]["mask_path"] == assets["masks"]["path"]
                and assets["bubbles"]["label"] == "anomalous" and row["mask_positive_pixels"] > 0,
                f"Invalid fixed-pose defect pairing: {sample}")
        stem = sample.split("/")[-1]
        for key, mode in (("bubbles", "RGB"), ("masks", "L")):
            path = V1_DATA / "regular" / key / f"{stem}.png"
            require(assets[key]["path"] == path.relative_to(V1_DATA).as_posix()
                    and sha(path) == assets[key]["sha256"], f"Unexpected or changed defect asset: {path}")
            with Image.open(path) as image:
                require(image.format == "PNG" and image.mode == mode and image.size == (FRAME, FRAME),
                        f"Invalid defect asset: {path}")
                if key == "masks":
                    levels = image.histogram()
                    require(not any(levels[1:-1]) and levels[-1] == row["mask_positive_pixels"],
                            f"Defect mask differs from audit: {path}")
        rows.append({**row, "files": {key: {**assets[key], "path": str(V1_DATA / assets[key]["path"])}
                                     for key in ("bubbles", "masks")}})
    return rows


def render_artifacts(path, rgb, mask, score, display_max, colormap):
    import numpy as np
    from PIL import Image

    slug = path.name
    rgb = Image.fromarray(rgb)
    rgb.save(path.with_name(f"{slug}_image.png"))
    heatmap = Image.fromarray(colormap(np.clip(score / display_max, 0, 1), bytes=True)[..., :3])
    heatmap.save(path.with_name(f"{slug}_heatmap.png"))
    Image.blend(rgb, heatmap, 0.45).save(path.with_name(f"{slug}_heatmap_overlay.png"))
    Image.fromarray(mask.astype(np.uint8) * 255).save(path.with_name(f"{slug}_ground_truth.png"))
    overlay = np.array(rgb)
    overlay[mask] = (0, 255, 140)
    Image.blend(rgb, Image.fromarray(overlay), 0.55).save(path.with_name(f"{slug}_ground_truth_overlay.png"))


def evaluate(stage, folder, normals, defects, runtime, threshold=None, display_max=None):
    import numpy as np
    from matplotlib import colormaps
    from phase3_metrics import PixelMetrics, RegionTargets

    raw = folder / "raw"
    raw.mkdir()
    visuals = folder / "visuals"
    visuals.mkdir()
    rows = []
    measurements = {}
    with runtime.measure("inference", measurements):
        for role, records, key in (("varied_normal", normals, "normal"),
                                   ("fixed_pose_defect", defects, "bubbles")):
            for record in records:
                sample = record["sample_id"].replace("/", "_")
                path = raw / f"{sample}.npy"
                rgb = runtime.load_rgb(record["files"][key])
                start = perf_counter()
                score, image_score = runtime.infer(rgb)
                seconds = perf_counter() - start
                require(score.shape == (FRAME, FRAME) and score.dtype == np.float32
                        and np.isfinite(score).all() and score.min() >= 0
                        and math.isfinite(image_score) and image_score >= 0, f"Invalid scores: {sample}")
                np.save(path, score, allow_pickle=False)
                rows.append({"id": record["sample_id"], "cohort": role, "image_score": float(image_score),
                             "map_max": float(score.max()), "inference_seconds": seconds,
                             "score_map": path.relative_to(folder).as_posix()})
                print(f"{stage}: {sample} score={image_score:.6f} ({seconds:.2f}s)", flush=True)
    if stage == "validation":
        threshold = max(row["image_score"] for row in rows if row["cohort"] == "varied_normal")
        display_max = max(max(row["map_max"] for row in rows), 1e-12)
    negatives, positives = rows[:len(normals)], rows[len(normals):]
    maximum = runtime.metric_max or max(max(row["map_max"] for row in positives), 1e-12)
    metrics = PixelMetrics(maximum)
    cmap = colormaps["magma"]
    for index, (row, record) in enumerate(zip(rows, normals + defects, strict=True)):
        row["flagged"] = row["image_score"] > threshold
        is_defect = index >= len(normals)
        if is_defect or index < 3:
            score = np.load(folder / row["score_map"], allow_pickle=False)
            if is_defect:
                mask = runtime.load_mask(record["files"]["masks"])
                require(mask.dtype == np.uint8 and np.isin(mask, (0, 255)).all()
                        and int(np.count_nonzero(mask)) == record["mask_positive_pixels"],
                        f"Mask differs from audit: {record['sample_id']}")
                mask = mask > 0
                metrics.update(score, RegionTargets(mask))
                row["positive_pixels"] = int(mask.sum())
            else:
                mask = np.zeros((FRAME, FRAME), dtype=bool)
            if not is_defect or index - len(normals) < 3:
                rgb = runtime.load_rgb(record["files"]["bubbles" if is_defect else "normal"])
                render_artifacts(visuals / row["id"].replace("/", "_"), rgb, mask, score, display_max, cmap)
    fp = sum(row["flagged"] for row in negatives)
    tp = sum(row["flagged"] for row in positives)
    with (folder / "scores.csv").open("w", encoding="utf-8", newline="") as output:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        table = csv.DictWriter(output, fields)
        table.writeheader()
        table.writerows(rows)
    save_chart(folder, negatives, positives, threshold, stage)
    return {
        "stage": stage, "image_auroc": None, "image_ap": None,
        "threshold": threshold, "threshold_rule": "Flag score > maximum V2 validation normal score",
        "display_max": display_max, "fixed_pose_pixel_score_max": maximum,
        "varied_normal": {"frames": len(negatives), "false_positives": fp, "false_positive_rate": fp / len(negatives)},
        "fixed_pose_defects": {"frames": len(positives), "true_positives": tp, "sensitivity": tp / len(positives),
                               **metrics.compute()},
        "measurements": measurements, "rows": rows,
    }


def save_chart(folder, normals, defects, threshold, stage):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    for index, (rows, label) in enumerate(((normals, "Varied normals"), (defects, "Fixed-pose defects"))):
        ax.scatter([row["image_score"] for row in rows], [index] * len(rows), alpha=0.65, label=label)
    ax.axvline(threshold, linestyle="--", color="black", label="Validation threshold")
    ax.set(yticks=[], xlabel="Raw image score", title=f"Regular lighting · {stage}")
    ax.legend()
    fig.savefig(folder / "image_scores.png", dpi=150)
    plt.close(fig)


def source_hashes(model_folder):
    sources = [Path(__file__), *model_folder.joinpath("V2").glob("*.py"),
               *ANOMALY.joinpath("patchcore", "V2").glob("*.py")]
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in sorted(set(sources))}


def cuda_environment(seed, fp32):
    import cv2
    import numpy as np
    import torch

    require(torch.cuda.is_available() and torch.version.cuda == "12.8", "Expected the existing CUDA 12.8 environment")
    cv2.setNumThreads(2)
    if fp32:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    gpu = torch.cuda.get_device_properties(0)
    versions = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu.name,
                "vram_mib_available": gpu.total_memory / 1024 ** 2}
    print(json.dumps(versions), flush=True)
    return versions


def runtime(load, infer, measure, configuration, measurements, metric_max=None):
    return SimpleNamespace(load_rgb=lambda asset: load(asset, "RGB"), load_mask=lambda asset: load(asset, "L"),
                           infer=infer, measure=measure, configuration=configuration, measurements=measurements,
                           metric_max=metric_max)


def run_cli(model_folder, model_name, build, validate_imports):
    parser = argparse.ArgumentParser(description=f"{model_name} regular-light V2 workflow")
    parser.add_argument("action", nargs="?", default="validate",
                        choices=("check", "check-env", "validate", "test", "all"))
    parser.add_argument("run", nargs="?", type=Path, help="Completed validation run folder for test")
    review = parser.add_mutually_exclusive_group()
    review.add_argument("--accept-near-duplicates", action="store_true",
                        help="Record acceptance of renderer-flagged pairs after manual review")
    review.add_argument("--use-reviewed-split", action="store_true",
                        help="Use the recorded decision for the fingerprinted V2 split")
    args = parser.parse_args()
    if (args.action == "test") != (args.run is not None):
        parser.error("Only test takes a run folder, which is required for test.")
    if args.action == "test" and (args.accept_near_duplicates or args.use_reviewed_split):
        parser.error("Test uses the near-duplicate decision recorded during validation.")
    results = model_folder / "results" / "V2"
    if args.action == "check-env":
        validate_imports()
        return
    prior = None
    if args.action == "test":
        out = args.run.resolve()
        require(out.parent == results.resolve(), f"Expected a validation run in {results}")
        require(not (out / "test").exists(), f"Test outputs already exist in {out / 'test'}")
        prior = read_json(out / "validation" / "report.json")
    near_accepted = (prior["near_duplicates_accepted"] if prior
                     else args.accept_near_duplicates or args.use_reviewed_split)
    normals, split, audit, manifest, state, hashes = preflight(near_accepted, args.use_reviewed_split)
    print(f"V2 audit verified: {COUNTS}; cross-split candidates: {len(state['split_audit']['cross_split_candidates'])}",
          flush=True)
    if args.action == "check":
        defect_records(split, audit, "validation")
        print("Verified the three fixed-pose validation image/mask pairs.", flush=True)
        return
    validation_defects = defect_records(split, audit, "validation") if not prior else None
    test_defects = defect_records(split, audit, "test") if args.action == "test" else None
    code_hashes = source_hashes(model_folder)
    if prior:
        require(prior["model"] == model_name and prior["source_sha256"] == code_hashes
                and prior["dataset_sha256"] == hashes, "Validation model, source code or dataset changed")
        validation_scores = {row["id"]: row["image_score"] for row in prior["rows"]
                             if row["cohort"] == "varied_normal"}
        require(set(validation_scores) == {row["sample_id"] for row in normals["validation"]}
                and all(math.isfinite(score) for score in validation_scores.values())
                and prior["threshold"] == max(validation_scores.values()), "Invalid saved validation calibration")
        checkpoint = out / "regular_state.pt"
        require(sha(checkpoint) == prior["checkpoint"]["sha256"], "Validation checkpoint changed")
    else:
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        out = results / f"regular_v2_{suffix}"
        out.mkdir(parents=True)
        checkpoint = out / "regular_state.pt"
        for original, key in ((DATA / "manifest_v2.json", "manifest_v2"), (DATA / "dataset.json", "dataset_v2")):
            snapshot = out / original.name
            shutil.copyfile(original, snapshot)
            require(sha(snapshot) == hashes[key], f"Dataset metadata changed during preflight: {original}")
    print(f"Run directory: {out}", flush=True)
    runtime = build(checkpoint, normals["train"] if not prior else None, validation_defects)
    if not prior:
        validation_dir = out / "validation"
        validation_dir.mkdir()
        prior = evaluate("validation", validation_dir, normals["validation"], validation_defects, runtime)
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False)
        prior.update({
            "model": model_name, "configuration": runtime.configuration, "fit_measurements": runtime.measurements,
            "checkpoint": {"sha256": sha(checkpoint), "file": checkpoint.name},
            "dataset_sha256": hashes, "source_sha256": code_hashes, "near_duplicates_accepted": near_accepted,
            "fold_counts": dict(Counter(row["split"] for row in manifest.values())),
            "python": sys.version, "platform": platform.platform(), "git_revision": git.stdout.strip(),
        })
        write_json(validation_dir / "report.json", prior)
        print(f"Completed validation: {validation_dir}", flush=True)
    if args.action in ("test", "all"):
        if args.action == "all":
            del runtime
            runtime = build(checkpoint, None, None)
        if test_defects is None:
            test_defects = defect_records(split, audit, "test")
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        test_dir = out / f"test_pending_{suffix}"
        test_dir.mkdir()
        result = evaluate("test", test_dir, normals["test"], test_defects,
                          runtime, prior["threshold"], prior["display_max"])
        result.update({"model": model_name, "checkpoint": prior["checkpoint"], "dataset_sha256": hashes,
                       "validation_report_sha256": sha(out / "validation" / "report.json")})
        write_json(test_dir / "report.json", result)
        test_dir.rename(out / "test")
        print(f"Completed test: {out / 'test'}", flush=True)
