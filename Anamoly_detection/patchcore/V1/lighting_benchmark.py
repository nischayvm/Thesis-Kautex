"""Complete the fixed-configuration PatchCore lighting validation benchmark."""

import csv
import hashlib
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from time import perf_counter

import phase3 as pipeline
import smoke
import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import colormaps
from sklearn.metrics import average_precision_score, roc_auc_score

from phase3_metrics import BINS, DIAGNOSTIC_FPR, FPR_LIMIT, SIZE_LABELS, PixelMetrics, RegionTargets
from smoke import FRAME, HERE, MIB, WEIGHTS, load_asset, measure, require
from split_dataset import SEED, SPLIT, load_audit, split_text

matplotlib.use("Agg")
from matplotlib import pyplot as plt

RED_RUN = HERE / "runs" / "red_phase3_20261124T075750_651308Z"
COLOURS = ("regular", "red", "green")
BUDGET = 128
STITCH = "gaussian"
LABELS = ("Regular", "Red", "Green")
PALETTE = ("#6d7386", "#ba504b", "#27856a")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_reference(report, split, preset):
    configuration = report["configuration"]
    expected = {
        "split_sha256": sha256(SPLIT), "seed": SEED, "fit_colour": "red",
        "fit_normal_ids": split["coloured_setup_ids"]["train"],
        "scored_validation_ids": split["coloured_setup_ids"]["validation"],
        "backbone": "wide_resnet50_2.IMAGENET1K_V2", "weight_sha256": sha256(WEIGHTS),
        "layers": list(smoke.LAYERS), "rgb_mean": preset.mean, "rgb_std": preset.std,
        "precision": "FP32; TF32 disabled", "batch_size": 1, "tile": smoke.TILE,
        "overlap": smoke.OVERLAP, "stride": smoke.STRIDE, "resize": None, "crop": None,
        "coreset_ratio": smoke.RATIO, "nearest_neighbors": smoke.NEIGHBORS,
        "query_chunk": smoke.QUERY_CHUNK, "bank_chunk": smoke.BANK_CHUNK,
        "gaussian_sigma_px": pipeline.SIGMA, "seam_control_offset_px": pipeline.SEAM_CONTROL_OFFSET,
        "metric_histogram_bins": BINS, "aupro_fpr_limit": FPR_LIMIT, "region_diagnostic_fpr": DIAGNOSTIC_FPR,
    }
    for field, current in expected.items():
        require(configuration[field] == current, f"Phase 3 reference mismatch for {field}")
    bank = report["banks"][str(BUDGET)]
    require(STITCH in configuration["stitch"] and bank["candidate_rows"] == 70 * 25 * BUDGET,
            "Selected Phase 3 configuration unavailable")
    shared = {key: value for key, value in expected.items() if key not in {"fit_colour", "fit_normal_ids", "scored_validation_ids"}}
    return {**shared, "candidate_budget_per_tile": BUDGET, "stitch": STITCH,
            "sampler": configuration["sampler"], "coreset": configuration["coreset"]}


def cohort(records, split, colour):
    if colour == "regular":
        regular = split["regular"]
        train = [records[regular["reference_sample_id"]]]
        require(train[0]["files"]["normal"]["path"] == regular["reference_normal"]
                and not regular["normal_validation"], "Unexpected fixed-reference split")
        valid = [records[key] for key in regular["defective_validation_ids"]]
        require(set(regular["defective_validation_ids"]).isdisjoint(regular["defective_test_ids"]),
                "Regular validation overlaps held-out defects")
    else:
        ids = split["coloured_setup_ids"]
        require(set(ids["train"] + ids["validation"]).isdisjoint(ids["test"]), "Test overlaps fit or validation")
        train = [records[f"{colour}/{key}"] for key in ids["train"]]
        valid = [records[f"{colour}/{key}"] for key in ids["validation"]]
    return train, valid


def describe_capture(train, validation):
    fields = ("camera_height_m", "led_height_m", "effective_led_height_m", "rig_intensity",
              "ceiling_base_intensity", "dark_field_mix")

    def summarize(records):
        setups = [record["render_setup"] for record in records]
        return {"views": len(setups), "distinct_setups": len({tuple(s.values()) for s in setups}),
                "camera": records[0]["camera"], "ranges": {key: [min(s[key] for s in setups), max(s[key] for s in setups)]
                                                          for key in fields},
                "partial_dark_field_views": sum(s["dark_field_mix"] > 0 for s in setups)}

    return {
        "training": summarize(train), "validation": summarize(validation),
        "validation_instance_totals": {key: sum(row[key] for row in validation)
                                       for key in ("flange_count", "cross_ply_count", "mask_positive_pixels")},
        "validation_defect_seeds": [row["defect_seed"] for row in validation],
    }


def defect_targets(record):
    mask = load_asset(record["files"]["masks"], "L")
    foreground = mask == 255
    require(np.all((mask == 0) | foreground) and foreground.sum() == record["mask_positive_pixels"],
            f"Invalid ground truth {record['sample_id']}")
    return mask, foreground


def score_validation(folder, predictions, validation):
    score_max = max(row["maps"][STITCH]["max"] for row in predictions)
    metric = PixelMetrics(score_max)
    records = {row["sample_id"]: row for row in validation}
    background = RegionTargets(np.zeros((FRAME, FRAME), dtype=bool))
    for row in predictions:
        targets = background
        if row["role"] == "bubbles":
            _, foreground = defect_targets(records[row["sample_id"]])
            targets = RegionTargets(foreground)
        metric.update(np.load(folder / row["maps"][STITCH]["path"], allow_pickle=False), targets)
    result = metric.compute()
    coarse = metric.compute(coarsen=2)
    result["half_bin_resolution_abs_difference"] = {key: abs(result[key] - coarse[key])
                                                    for key in ("pixel_auroc", "pixel_ap", "aupro")}
    result["mean_inference_s"] = float(np.mean([row["tile_inference_s"] + row["maps"][STITCH]["stitch_s"]
                                              for row in predictions]))
    result["image_auroc"] = result["image_ap"] = result["normal_seam"] = None
    normals = [row["maps"][STITCH] for row in predictions if row["role"] == "normal"]
    if normals:
        labels = [row["role"] == "bubbles" for row in predictions]
        scores = [row["score"] for row in predictions]
        result["image_auroc"] = float(roc_auc_score(labels, scores))
        result["image_ap"] = float(average_precision_score(labels, scores))
        seam = float(np.mean([entry["seam_step"] for entry in normals]))
        control = float(np.mean([entry["control_step"] for entry in normals]))
        result["normal_seam"] = {"step": seam, "control_step": control, "ratio": seam / control}
    return result, score_max


def save_model(model, path):
    torch.save(model.state_dict(), path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(torch.equal(saved["memory_bank"], model.memory_bank.cpu()), f"Checkpoint bank mismatch for {path.name}")
    return {"path": path.relative_to(path.parent.parent).as_posix(), "sha256": sha256(path), "dimensions": list(model.memory_bank.shape),
            "bank_mib": model.memory_bank.nbytes / MIB}


def fit_and_evaluate(colour, train, validation, folder, stitcher, mean, std):
    (folder / "raw").mkdir(parents=True, exist_ok=False)
    stats = {}
    with measure(f"{colour}_load_backbone", stats):
        model = smoke.make_model(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    with torch.inference_mode():
        with measure(f"{colour}_candidate_extraction", stats):
            primary, secondary = pipeline.collect_candidates(model, train, mean, std)
        candidate_count = 2 * len(primary)
        torch.cuda.empty_cache()
        with measure(f"{colour}_coreset", stats):
            model.memory_bank = pipeline.fit_bank(primary, secondary, BUDGET)
        del primary, secondary
        checkpoint = save_model(model, folder / "fitted_state.pt")
        torch.cuda.empty_cache()
        with measure(f"{colour}_validation_inference", stats):
            predictions = pipeline.evaluate(model, validation, BUDGET, stitcher, folder, mean, std,
                                            stitches=(STITCH,), include_normal=(colour != "regular"))
    del model
    torch.cuda.empty_cache()
    start = perf_counter()
    metrics, score_max = score_validation(folder, predictions, validation)
    result = {"train_ids": [r["sample_id"] for r in train], "validation_ids": [r["sample_id"] for r in validation],
              "validation_normal_count": sum(row["role"] == "normal" for row in predictions),
              "candidate_rows": candidate_count, "checkpoint": checkpoint, "measurements": stats,
              "metrics_seconds": perf_counter() - start, "metric_score_max": score_max,
              "metrics": metrics, "predictions": predictions, "capture": describe_capture(train, validation)}
    (folder / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def package_red(folder, report, train, validation):
    bank_path = RED_RUN / f"memory_bank_{BUDGET}.pt"
    with torch.inference_mode():
        model = smoke.make_model(torch.load(WEIGHTS, map_location="cpu", weights_only=True)).eval()
        bank = torch.load(bank_path, map_location="cpu", weights_only=True)
        require(list(bank.shape) == report["banks"][str(BUDGET)]["dimensions"] and bank.dtype == torch.float32,
                "Red fitted bank does not match its report")
        model.memory_bank = bank.cuda()
        checkpoint = save_model(model, folder / "fitted_state.pt")
    del model, bank
    torch.cuda.empty_cache()
    return {"source_report": (RED_RUN / "report.json").relative_to(HERE).as_posix(),
            "source_bank_sha256": sha256(bank_path), "train_ids": [r["sample_id"] for r in train],
            "validation_ids": [r["sample_id"] for r in validation], "validation_normal_count": len(validation),
            "candidate_rows": report["banks"][str(BUDGET)]["candidate_rows"], "checkpoint": checkpoint,
            "measurements": report["measurements"], "metric_score_max": report["configuration"]["display_range"][1],
            "metrics": report["metrics"][f"{BUDGET}_{STITCH}"], "predictions": report["per_image"][str(BUDGET)],
            "capture": describe_capture(train, validation)}


def save_comparable_images(out, results, cohorts, examples):
    maximum = max(row["maps"][STITCH]["max"] for result in results.values() for row in result["predictions"])
    require(maximum > 0 and np.isfinite(maximum), "Invalid common display scale")
    colormap = colormaps["inferno"]
    sheet = Image.new("RGB", (1656, 928), "#1a1a1a")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), f"Defective validation renders and score maps | shared display scale: 0 to {maximum:.2f}", fill="white")
    for row_index, colour in enumerate(COLOURS):
        folder = out / colour / "visuals"
        folder.mkdir()
        validation = {r["sample_id"]: r for r in cohorts[colour][1]}
        for row in results[colour]["predictions"]:
            if row["role"] != "bubbles":
                continue
            key = row["sample_id"]
            sample = key.split("/")[-1]
            overlay_file = folder / f"{sample}_ground_truth_overlay.png"
            score_root = RED_RUN if colour == "red" else folder.parent
            if colour == "red":
                shutil.copyfile(RED_RUN / "visuals" / overlay_file.name, overlay_file)
            else:
                mask, foreground = defect_targets(validation[key])
                rgb = load_asset(validation[key]["files"]["bubbles"], "RGB")
                rgb[foreground] = np.round(0.45 * rgb[foreground] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
                Image.fromarray(rgb).save(overlay_file)
                Image.fromarray(mask).save(folder / f"{sample}_mask.png")
            score = np.load(score_root / row["maps"][STITCH]["path"], allow_pickle=False)
            require(score.shape == (FRAME, FRAME) and score.dtype == np.float32 and np.isfinite(score).all(),
                    f"Invalid saved raw map for {key}")
            heatmap_file = folder / f"{sample}_heatmap.png"
            Image.fromarray(colormap(np.clip(score / maximum, 0, 1), bytes=True)[..., :3]).save(heatmap_file)
            if sample in examples[colour]:
                index = examples[colour].index(sample)
                y = 35 + row_index * 294
                for offset, (label, path) in enumerate((("ground truth", overlay_file), ("heatmap", heatmap_file))):
                    x = 8 + (2 * index + offset) * 275
                    draw.text((x, y), f"{key} · {label}", fill="white")
                    with Image.open(path) as image:
                        sheet.paste(image.resize((265, 265), Image.Resampling.LANCZOS), (x, y + 18))
    sheet.save(out / "lighting_heatmaps.png")
    return [0, maximum]


def save_comparison(out, results):
    fields = ("pixel_auroc", "pixel_ap", "aupro", "image_auroc", "image_ap")
    headings = ("Pixel AUROC", "Pixel AP", "AUPRO at 30% FPR", "Image AUROC", "Image AP")
    with (out / "comparison.csv").open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("lighting", "training_normals", "validation_normals", "validation_defects", *fields,
                         "bank_mib", "mean_inference_s"))
        for colour in COLOURS:
            row = results[colour]
            metrics = row["metrics"]
            writer.writerow((colour, len(row["train_ids"]), row["validation_normal_count"], len(row["validation_ids"]),
                             *(metrics[key] if metrics[key] is not None else "N/A" for key in fields),
                             row["checkpoint"]["bank_mib"], metrics["mean_inference_s"]))

    save_charts(out, results, fields, headings)


def save_charts(out, results, fields, headings):
    fig, axes = plt.subplots(1, len(fields), figsize=(13, 3.3), sharey=True, layout="constrained")
    for ax, field, title in zip(axes, fields, headings, strict=True):
        for slot, colour in enumerate(COLOURS):
            value = results[colour]["metrics"][field]
            if value is None:
                ax.text(slot, 0.045, "N/A", ha="center", color=PALETTE[slot])
                continue
            ax.bar(slot, value, color=PALETTE[slot], width=0.64)
            ax.text(slot, value + 0.022, f"{value:.3f}", ha="center", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(LABELS)), LABELS, fontsize=8)
        ax.set_xlim(-0.5, len(LABELS) - 0.5)
        ax.set_ylim(0, 1.10)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("PatchCore lighting comparison · validation only")
    fig.savefig(out / "validation_metrics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True, layout="constrained")
    for ax, field, title in zip(axes, ("aupro", "mean_overlap_at_1pct_fpr"),
                                ("AUPRO at 30% FPR", "Mean overlap at 1% FPR"), strict=True):
        for index, colour in enumerate(COLOURS):
            groups = results[colour]["metrics"]["size_specific"]
            values = [groups[label][field] if groups[label][field] is not None else np.nan for label in SIZE_LABELS]
            ax.bar(np.arange(3) + (index - 1) * 0.24, values, width=0.22, color=PALETTE[index], label=LABELS[index])
        ax.set_xticks(range(3), ("<64", "64–511", "≥512"))
        ax.set_xlabel("Visible component area (px²)")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=3, frameon=False)
    fig.savefig(out / "defect_size_metrics.png", dpi=180)
    plt.close(fig)


def main():
    cv2.setNumThreads(2)
    require(torch.cuda.is_available() and torch.version.cuda == "12.8", "Expected the Phase 3 CUDA environment")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    records, audit_digest = load_audit()
    versioned = SPLIT.read_text(encoding="utf-8")
    require(versioned == split_text(records, audit_digest), "Frozen split differs from the original assignment")
    split = json.loads(versioned)
    reference_path = RED_RUN / "report.json"
    red_hash = sha256(reference_path)
    red = json.loads(reference_path.read_text(encoding="utf-8"))
    preset = smoke.Wide_ResNet50_2_Weights.IMAGENET1K_V2.transforms()
    configuration = verify_reference(red, split, preset)
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "torchvision": version("torchvision"),
                   "anomalib": version("anomalib"), "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                   "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / MIB}
    require(all(red["environment"][key] == value for key, value in environment.items()),
            "Environment differs from Phase 3")
    cohorts = {colour: cohort(records, split, colour) for colour in COLOURS}
    require(all(r["render_setup"] == g["render_setup"] and r["defect_seed"] != g["defect_seed"]
                and r["defect_instances"] != g["defect_instances"]
                for red_group, green_group in zip(cohorts["red"], cohorts["green"], strict=True)
                for r, g in zip(red_group, green_group, strict=True)), "Colored capture pairing differs from the audit")
    out = HERE / "runs" / datetime.now(timezone.utc).strftime("lighting_benchmark_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(SPLIT, out / SPLIT.name)
    print(f"Run directory {out}", flush=True)
    mean = torch.tensor(preset.mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(preset.std, device="cuda").view(1, 3, 1, 1)
    stitcher = pipeline.Stitcher()
    results = {}
    for colour in ("regular", "green"):
        results[colour] = fit_and_evaluate(colour, *cohorts[colour], out / colour, stitcher, mean, std)
    folder = out / "red"
    folder.mkdir()
    results["red"] = package_red(folder, red, *cohorts["red"])
    examples = [key.split("/")[-1] for key in red["configuration"]["display_examples"]]
    report = {
        "configuration": configuration, "red_reference_report_sha256": red_hash, "environment": environment,
        "git_base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "script_sha256": {name: sha256(HERE / name) for name in
                          ("lighting_benchmark.py", "phase3.py", "phase3_metrics.py", "smoke.py", "split_dataset.py")},
        "lighting": results,
    }
    save_comparison(out, results)
    report["common_display_range"] = save_comparable_images(out, results, cohorts,
                                                            {"red": examples, "green": examples, "regular":
                                                             [key.split("/")[-1] for key in split["regular"]["defective_validation_ids"]]})
    require(sha256(reference_path) == red_hash and sha256(SPLIT) == configuration["split_sha256"],
            "Historical report or frozen split changed")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved consolidated results to {out}", flush=True)


if __name__ == "__main__":
    main()
