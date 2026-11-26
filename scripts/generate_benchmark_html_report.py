"""Build an offline, versioned report from saved ANNAlyzer results."""

import argparse
import hashlib
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean
from urllib.parse import quote

import numpy as np
from matplotlib import colormaps
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(__file__).with_name("benchmark_report_template.html")
MODEL_NAMES = {"patchcore": "PatchCore", "anomalydino": "AnomalyDINO", "unet": "U-Net", "flownomaly": "Flownomaly"}
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
METRICS = ("pixel_auroc", "pixel_ap", "aupro", "image_auroc", "image_ap")
PICTURES = {
    "image": ("image", "original"),
    "mask": ("ground_truth", "mask"),
    "truthOverlay": ("ground_truth_overlay",),
    "heat": ("heatmap", "probability"),
    "overlay": ("heatmap_overlay", "prediction_overlay"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def sha(path):
    if not path.is_file():
        return None
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def existing(base, relative):
    if not relative or not isinstance(relative, str):
        return None
    path = (base / relative).resolve()
    return path if path.is_relative_to(base.resolve()) and path.is_file() else None


def maximum(values, *keys):
    return max((row[key] for row in values for key in keys if isinstance(row.get(key), (int, float))), default=None)


def performance(metrics, numbers, configuration, checkpoint):
    timings = list(numbers.values())
    bank = configuration.get("bank_dimensions") or checkpoint.get("dimensions")
    return {
        "meanSeconds": metrics.get("mean_inference_s"),
        "ram": maximum(timings, "rss_peak_mib", "rss_peak_mib_sampled"),
        "allocated": maximum(timings, "cuda_peak_allocated_mib"),
        "reserved": maximum(timings, "cuda_peak_reserved_mib"),
        "bank": bank,
        "bankMib": configuration.get("bank_mib", checkpoint.get("bank_mib")),
        "timings": {key: row.get("seconds") for key, row in numbers.items()},
    }


def metric_definition(configuration):
    limit = configuration.get("aupro_fpr_limit", 0.30)
    bins = configuration.get("metric_histogram_bins", configuration.get("histogram_bins", 65_536))
    frame = configuration.get("frame", 2048)
    return {"metricDefinition": f"native:{frame}:{limit}:{bins}", "auproLimit": limit}


def save_preview(image, path, quality):
    pending = path.with_suffix(f".{os.getpid()}.pending")
    try:
        image.save(pending, format="JPEG", quality=quality)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


class Report:
    def __init__(self, workspace, output):
        self.root = workspace.resolve()
        self.anomaly = self.root / "Anamoly_detection"
        self.output = output.resolve()
        self.media = self.output / "media"
        self.media.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.anomaly / "dataset_audit_manifest_v1.jsonl"
        self.audit = {}
        if self.audit_path.is_file():
            for line in self.audit_path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self.audit[row["sample_id"]] = row["files"]

    def link(self, path):
        if path is None or not path.is_file():
            return None
        try:
            return quote(Path(os.path.relpath(path, self.output)).as_posix(), safe="/")
        except ValueError:
            return path.resolve().as_uri()

    def cached_media(self, name, *sources):
        stamp = [name, *(str(p) + str(p.stat().st_size) + str(p.stat().st_mtime_ns) for p in sources)]
        key = hashlib.sha256("\n".join(stamp).encode()).hexdigest()[:24]
        return self.media / f"{key}.jpg"

    def picture(self, path):
        url = self.link(path)
        if url is None:
            return None
        thumb = self.cached_media("thumb-400", path)
        if not thumb.is_file():
            with Image.open(path) as image:
                image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                save_preview(image.convert("RGB"), thumb, quality=85)
        return {"url": url, "thumb": self.link(thumb), "name": path.name}

    def sources(self, sample, normal, manifest, audit_hash=None):
        if sample in manifest and "files" in manifest[sample]:
            row = manifest[sample]
            image_root = Path(row["dataset"])
            image = row["files"]["normal" if normal else "bubbles"]
            mask = None if normal else row["files"]["masks"]
        elif normal and sample in manifest:
            row = manifest[sample]
            image_root, image, mask = self.root, row, None
        elif audit_hash is None or audit_hash == sha(self.audit_path):
            row = self.audit.get(sample, {})
            image_root = self.root / "data/annalyzer_batch_2048"
            image = row.get("normal" if normal else "bubbles")
            mask = None if normal else row.get("masks")
        else:
            return None, None, None
        rgb = existing(image_root, image.get("path")) if image else None
        ground_truth = existing(image_root, mask.get("path")) if mask else None
        image_hash = image.get("sha256") if image else None
        mask_hash = mask.get("sha256") if mask else None
        if rgb is not None and (not image_hash or sha(rgb) != image_hash):
            rgb = None
        if ground_truth is not None and (not mask_hash or sha(ground_truth) != mask_hash):
            ground_truth = None
        if rgb and (normal or ground_truth):
            return rgb, ground_truth, f"{image_hash}:{mask_hash or 'normal'}"
        return rgb, ground_truth, None

    def display_maps(self, images, raw, rgb, scale, palette):
        if raw is None or rgb is None or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            return
        if images["heat"] and images["overlay"]:
            return
        heat_path = self.cached_media(f"heat-{scale}-{palette}", raw)
        overlay_path = self.cached_media(f"overlay-{scale}-{palette}", raw, rgb)
        with Image.open(rgb) as original:
            if not heat_path.is_file() or (not images["overlay"] and not overlay_path.is_file()):
                values = np.load(raw, mmap_mode="r", allow_pickle=False)
                if values.ndim != 2 or values.shape != (original.height, original.width):
                    raise ValueError(f"Score map does not align with its source image: {raw}")
                if not np.isfinite(values).all():
                    raise ValueError(f"Score map contains non-finite values: {raw}")
                array = colormaps[palette](np.clip(values / scale, 0, 1), bytes=True)[..., :3]
                colored = Image.fromarray(array)
                if not images["heat"] and not heat_path.is_file():
                    save_preview(colored, heat_path, quality=94)
                if not images["overlay"] and not overlay_path.is_file():
                    save_preview(Image.blend(original.convert("RGB"), colored, 0.45), overlay_path, quality=94)
        for key, path in (("heat", heat_path), ("overlay", overlay_path)):
            if not images[key]:
                images[key] = self.picture(path)
                if images[key]:
                    images[key]["generated"] = True

    def frame(self, sample, normal, score, seconds, flagged, raw, visuals, stem, scale, palette, manifest, audit_hash):
        rgb, mask, identity = self.sources(sample, normal, manifest, audit_hash)
        images = {}
        for key, suffixes in PICTURES.items():
            if key in ("mask", "truthOverlay") and normal:
                images[key] = None
                continue
            saved = next((p for suffix in suffixes if (p := visuals / f"{stem}_{suffix}.png").is_file()), None)
            source = {"image": rgb, "mask": mask}.get(key)
            images[key] = self.picture(source or saved)
        self.display_maps(images, raw, rgb, scale, palette)
        return {"sample": sample, "normal": normal, "identity": identity, "images": images,
                "score": score, "seconds": seconds, "flagged": flagged, "raw": self.link(raw)}

    def v1_groups(self, run, report_path, info):
        groups = []
        configuration = info.get("configuration", {})
        if not isinstance(info.get("lighting"), dict):
            return groups
        display = info.get("common_display_range") or info.get("shared_heatmap_display_scale") or [0, 1]
        split_path = self.anomaly / "patchcore/V1/split_v1.json"
        expected_split = configuration.get("split_sha256")
        audit_hash = (read_json(split_path).get("audit_manifest_sha256", "unverified")
                      if expected_split and sha(split_path) == expected_split else "unverified")
        for light, result in info["lighting"].items():
            if not isinstance(result, dict) or not isinstance(result.get("predictions"), list):
                continue
            folder = report_path.parent / light
            # The consolidated red PatchCore report references the preceding ablation's raw maps.
            red_source = result.get("source_report")
            other_raw = run.parent / Path(red_source).parent.name if isinstance(red_source, str) else folder
            frames = []
            for row in result["predictions"]:
                sample, role = row["sample_id"], row["role"]
                native = row.get("probability_path") or row.get("maps", {}).get("gaussian", {}).get("path")
                raw = existing(folder, native) or existing(other_raw, native)
                stem = sample.split("/")[-1]
                if role == "normal":
                    stem += "_normal"
                frame = self.frame(sample, role == "normal", row.get("image_score", row.get("score")),
                                   row.get("inference_s", row.get("tile_inference_s")), None, raw, folder / "visuals",
                                   stem, display[1], "inferno", {}, audit_hash)
                frames.append(frame)
            checkpoint = result.get("checkpoint", {})
            resource = dict(result.get("measurements", {}))
            for epoch in result.get("history", []):
                resource.update({f"Epoch {epoch['epoch']} · {key}": row for key, row in epoch["measurements"].items()})
            path = checkpoint.get("path") or result.get("best_checkpoint") or result.get("source_checkpoint")
            protocol = result.get("protocol") or (
                f"Normal-only fitting · {len(result.get('train_ids', []))} training images"
            )
            metrics = result.get("metrics", {})
            groups.append({
                "stage": "validation", "light": light, "protocol": protocol, "frames": frames,
                "pixelScope": "all", **metric_definition(configuration),
                "metrics": {key: metrics.get(key) for key in METRICS}, "sizes": metrics.get("size_specific", {}),
                "resources": performance(metrics, resource, configuration, checkpoint), "threshold": None,
                "thresholdRule": None,
                "imageRankingNote": ("This cohort has defective frames only."
                                     if frames and all(not frame["normal"] for frame in frames) else None),
                "report": self.link(report_path), "checkpoint": self.link(existing(report_path.parent, path)),
                "cohorts": None, "displayScale": [0, display[1]],
            })
        return groups

    def modern_group(self, run, report_path, data):
        stage = report_path.parent.name
        if stage not in ("validation", "test") or data.get("stage") != stage:
            return None
        paired = "normal_cohort" in data and "defect_cohort" in data
        normal_key, defect_key = ("normal_cohort", "defect_cohort") if paired else ("varied_normal", "fixed_pose_defects")
        if not all(key in data for key in ("rows", normal_key, defect_key)):
            return None
        if not isinstance(data["rows"], list) or any(
            row.get("cohort") not in ("matched_normal", "matched_defect", "varied_normal", "fixed_pose_defect")
            for row in data["rows"]
        ):
            return None
        validation = run / "validation/report.json"
        metadata = read_json(validation) if stage == "test" and validation.is_file() else data
        config = metadata.get("configuration", {})
        hashes = data.get("dataset_sha256", {})
        records = {}
        if hashes.get("workstation_split") and sha(run / "split.json") == hashes["workstation_split"]:
            split = read_json(run / "split.json")
            records.update((key, {**row, "dataset": split["dataset"]}) for key, row in split["records"].items())
        for manifest, digest in hashes.items():
            if re.fullmatch(r"manifest_v[1-9][0-9]*", manifest) and sha(run / f"{manifest}.json") == digest:
                records.update((row["image_id"], row) for row in read_json(run / f"{manifest}.json").values())
        frames = []
        for row in data["rows"]:
            sample = row["id"]
            normal = row["cohort"] in ("varied_normal", "matched_normal")
            audit_hash = hashes.get("audit_v1", "unverified")
            if normal and not paired and sample not in records:
                audit_hash = "unverified"
            frames.append(self.frame(sample, normal, row.get("image_score"), row.get("inference_seconds"),
                                     row.get("flagged"), existing(report_path.parent, row.get("score_map")),
                                     report_path.parent / "visuals", row.get("visual_stem", sample.replace("/", "_")),
                                     data.get("display_max"),
                                     "magma", records, audit_hash))
        defects, normal = data[defect_key], data[normal_key]
        metrics = {key: (defects if key.startswith("pixel_") or key == "aupro" else data).get(key) for key in METRICS}
        measurements = {**data.get("fit_measurements", {}), **data.get("measurements", {})}
        checkpoint = data.get("checkpoint", {})
        times = [frame["seconds"] for frame in frames if frame["seconds"] is not None]
        performance_data = performance({"mean_inference_s": mean(times) if times else None}, measurements, config, checkpoint)
        return {
            "stage": stage, "light": data.get("lighting", "regular"), "protocol": config.get("protocol", "Lighting evaluation"),
            "frames": frames, "pixelScope": data.get("pixel_scope", "defects"), **metric_definition(config), "metrics": metrics,
            "sizes": defects.get("size_specific", {}), "resources": performance_data,
            "threshold": data.get("threshold"), "cohorts": {"normals": normal, "defects": defects},
            "thresholdRule": data.get("threshold_rule"),
            "imageRankingNote": (f"{normal['frames']} varied clean frames and {defects['frames']} fixed-pose "
                                 "defective frames use different capture setups." if not paired else None),
            "report": self.link(report_path), "checkpoint": self.link(existing(run, checkpoint.get("file"))),
            "displayScale": [0, data.get("display_max")],
        }

    def run(self, run, version, model):
        report_paths = [path for path in (run / "report.json", run / "validation/report.json", run / "test/report.json")
                        if path.is_file()]
        completion = run / "complete.json"
        if completion.is_file():
            lights = read_json(completion).get("lights", [])
            for light in lights:
                for stage in ("validation", "test"):
                    path = existing(run, f"{light}/{stage}/report.json")
                    if path:
                        report_paths.append(path)
        if not report_paths:
            return None
        groups, configurations, warnings, ablations = [], {}, [], {}
        if completion.is_file():
            configurations["complete.json"] = read_json(completion)
        for path in report_paths:
            try:
                data = read_json(path)
                result = self.v1_groups(run, path, data) if path == run / "report.json" else []
                if path != run / "report.json":
                    group = self.modern_group(run, path, data)
                    result = [group] if group else []
                if isinstance(data.get("banks"), dict) and isinstance(data.get("per_image"), dict):
                    ablations = data.get("metrics", {})
                elif not result and not data.get("observed"):
                    warnings.append(f"{path.relative_to(run).as_posix()}: artifact view; no compatible frame table")
                groups.extend(result)
                configurations[path.relative_to(run).as_posix()] = {
                    key: data[key] for key in ("configuration", "environment", "config", "observed", "dataset_sha256",
                                               "git_revision", "git_base", "fold_counts", "banks", "capacity_estimates")
                    if key in data
                }
            except (OSError, ValueError, KeyError, TypeError) as error:
                warnings.append(f"{path.relative_to(run).as_posix()}: {error}")
        files = sorted(path for path in run.rglob("*") if path.is_file()
                       and not any(part.startswith(("test_pending_", ".")) for part in path.relative_to(run).parts))
        gallery = []
        for path in files:
            if path.suffix.lower() in IMAGE_TYPES:
                try:
                    gallery.append({**self.picture(path), "name": path.relative_to(run).as_posix()})
                except OSError as error:
                    warnings.append(f"{path.name}: {error}")
        return {"name": run.name, "version": version, "model": model, "groups": groups, "warnings": warnings,
                "configuration": configurations, "gallery": gallery,
                "ablations": ablations,
                "files": [{"url": self.link(path), "name": path.relative_to(run).as_posix()} for path in files],
                "key": f"{version}/{model}/{run.name}"}

    def collect(self):
        versions = {}
        for folder in sorted(self.anomaly.glob("*/results/V*")):
            if not folder.is_dir() or not re.fullmatch(r"V[1-9][0-9]*", folder.name):
                continue
            version = folder.name
            model = MODEL_NAMES.get(folder.parent.parent.name, folder.parent.parent.name)
            runs = [record for run in sorted(folder.iterdir(), reverse=True) if run.is_dir()
                    and (record := self.run(run, version, model)) is not None]
            if runs:
                print(f"{version} · {model}: {len(runs)} saved runs", flush=True)
                versions.setdefault(version, {})[model] = sorted(runs, key=lambda run: bool(run["groups"]), reverse=True)
        if not versions:
            raise ValueError(f"No completed result reports were found under {self.anomaly}")
        order = {label: index for index, label in enumerate(MODEL_NAMES.values())}
        result = []
        for name in sorted(versions, key=lambda version: int(version[1:])):
            models = versions[name]
            ranked = sorted(models, key=lambda model: (order.get(model, len(order)), model))
            result.append({"name": name, "models": {model: models[model] for model in ranked}})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "Anamoly_detection/reports",
                        help="Dedicated report output directory")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if any(output.is_relative_to(path) for path in [ROOT / "data", ROOT / "V1", ROOT / "synthetic_dataset", ROOT / ".git",
                                                   *ROOT.glob("Anamoly_detection/*/results")]):
        parser.error("Choose a report directory outside dataset, model results and backup folders.")
    report = Report(ROOT, output)
    versions = report.collect()
    content = json.dumps(versions, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8").replace("<!-- REPORT_DATA -->", content)
    page = output / "index.html"
    pending = page.with_suffix(f".{os.getpid()}.pending")
    try:
        pending.write_text(html, encoding="utf-8")
        pending.replace(page)
    finally:
        pending.unlink(missing_ok=True)
    print(f"Report: {page}")


if __name__ == "__main__":
    main()
