"""Render 99 additional regular-light normal views without altering the v1 dataset."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_annalyzer_batch import (CAMERA, COLORS, SIDE, SAMPLES, THRESHOLD, atomic_json,
                                    check_png, configure_lighting, configure_render, output_lock, render)


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "data" / "annalyzer_batch_2048"
AUDIT = ROOT / "Anamoly_detection" / "dataset_audit_manifest_v1.jsonl"
SPLIT = ROOT / "Anamoly_detection" / "patchcore" / "split_v1.json"
COUNT = 99
NEAR_DUPLICATE_RMSE = 2.5


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan():
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if split["audit_manifest_sha256"] != sha256(AUDIT):
        raise ValueError("The v1 audit manifest no longer matches the frozen split.")
    audit = {row["sample_id"]: row for row in
             map(json.loads, AUDIT.read_text(encoding="utf-8").splitlines())}
    fixed = audit[split["regular"]["reference_sample_id"]]
    fixed_normal = fixed["files"]["normal"]
    anchor = V1 / fixed_normal["path"]
    check_png(anchor)
    if sha256(anchor) != fixed_normal["sha256"]:
        raise ValueError("The v1 regular reference no longer matches its audit hash.")
    reference = {
        "image_id": "regular-v2/reference", "capture_setup_id": fixed["capture_setup_id"],
        "group_id": fixed["capture_setup_id"], "split": "train", "label": "normal",
        "lighting": "regular", "mask_path": None, "defect_seed": None, "source": "v1_reference",
        "camera": fixed["camera"], "underlying_scene": fixed["underlying_scene"],
        "part_design": fixed["part_design"], "render_setup": fixed["render_setup"],
        "path": anchor.relative_to(ROOT).as_posix(), "sha256": fixed_normal["sha256"],
    }
    folds = split["coloured_setup_ids"]
    omitted = max(folds["train"])
    views = {}
    for fold, ids in folds.items():
        for sample in sorted(ids):
            if sample == omitted:
                continue  # The existing fixed reference occupies one training slot.
            red, green = (audit[f"{color}/{sample}"] for color in ("red", "green"))
            if red["capture_setup_id"] != green["capture_setup_id"] or red["render_setup"] != green["render_setup"]:
                raise ValueError(f"Frozen color setup differs for {sample}.")
            setup = red["render_setup"]
            views[sample] = {
                "image_id": f"regular-v2/{sample}", "capture_setup_id": f"regular-v2/{sample}",
                "group_id": red["capture_setup_id"], "related_v1_ids": [red["sample_id"], green["sample_id"]],
                "split": fold, "label": "normal", "lighting": "regular", "mask_path": None,
                "defect_seed": None, "source": "v2_render", "camera": red["camera"],
                "underlying_scene": red["underlying_scene"], "part_design": red["part_design"],
                "render_setup": {
                    "height_band": setup["height_band"], "camera_height_m": setup["camera_height_m"],
                    "led_height_m": setup["led_height_m"], "rig_intensity": 1.,
                    "ceiling_base_intensity": 1., "dark_field_mix": 0.,
                },
            }
    heights = {(v["render_setup"]["camera_height_m"], v["render_setup"]["led_height_m"])
               for v in views.values()}
    if len(views) != COUNT or len(heights) != COUNT:
        raise ValueError("Expected exactly 99 distinct camera and LED height combinations.")
    return reference, views, omitted


def inspect_pixels(path):
    check_png(path)
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        full = np.empty(SIDE * SIDE * 4, dtype="<f4")
        image.pixels.foreach_get(full)
        rgb = np.ascontiguousarray(full.reshape(-1, 4)[:, :3])
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        image.scale(64, 64)
        small = np.empty(64 * 64 * 4, dtype=np.float32)
        image.pixels.foreach_get(small)
        return digest, small.reshape(-1, 4)[:, :3].copy()
    finally:
        bpy.data.images.remove(image)


def audit_split(records):
    seen = {}
    thumbnails = {}
    for key, record in records.items():
        digest, thumbnail = inspect_pixels(ROOT / record["path"])
        if digest in seen:
            raise ValueError(f"Identical decoded normal images: {seen[digest]} and {key}")
        seen[digest] = key
        thumbnails[key] = thumbnail
        record["blender_rgb_float32_sha256"] = digest
    candidates = []
    nearest = None
    for index, (key, record) in enumerate(records.items()):
        for other in list(records)[index + 1:]:
            if record["split"] == records[other]["split"]:
                continue
            rmse = float(np.sqrt(np.mean((thumbnails[key] - thumbnails[other])**2)) * 255.)
            pair = {"images": [key, other], "rmse_0_255": rmse}
            if nearest is None or rmse < nearest["rmse_0_255"]:
                nearest = pair
            if rmse <= NEAR_DUPLICATE_RMSE:
                candidates.append(pair)
    return {"exact_duplicates": 0, "nearest_cross_split_pair": nearest,
            "near_duplicate_threshold_0_255": NEAR_DUPLICATE_RMSE,
            "cross_split_candidates": sorted(candidates, key=lambda pair: pair["rmse_0_255"]),
            "split_review_required": bool(candidates),
            "method": "64 x 64 RGB thumbnail RMSE on Blender-decoded pixels; heuristic screen."}


def run(output, resume):
    if not bpy.app.background:
        raise RuntimeError("Launch this script in background Blender.")
    blend = Path(bpy.data.filepath)
    v1_state = json.loads((V1 / "dataset.json").read_text(encoding="utf-8"))
    v1_hashes = v1_state["configuration"]["source_sha256"]
    if (not v1_state["complete"] or not blend.is_file()
            or sha256(blend) != v1_hashes.get(blend.name)):
        raise ValueError("The batch requires the original completed v1 inspection scene.")
    reference, views, omitted = plan()
    output.relative_to(ROOT)
    state_file = output / "dataset.json"
    catalog_file = output / "manifest_v2.json"
    source_files = (blend, Path(__file__), SPLIT, AUDIT, Path(__file__).with_name("render_annalyzer_batch.py"),
                    Path(__file__).with_name("render_prototype_dataset.py"))
    config = {
        "schema": 2, "new_renders": COUNT, "seed": 24117, "resolution": [SIDE, SIDE], "samples": SAMPLES,
        "noise_threshold": THRESHOLD, "denoising": True, "device": "OptiX", "encoding": "8-bit RGB PNG",
        "rig_rgb": list(COLORS["regular"]), "omitted_v1_train_setup": omitted,
        "file_paths_relative_to": "project_root", "counts_including_reference": {"train": 70, "validation": 15, "test": 15},
        "source_sha256": {path.name: sha256(path) for path in source_files},
    }
    with output_lock(output):
        if resume:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            if state["configuration"] != config:
                raise ValueError("Resume inputs differ from the original v2 run.")
        else:
            if any(path.name != ".batch.lock" for path in output.iterdir()):
                raise FileExistsError(f"Use a new output folder: {output}")
            state = {"configuration": config, "complete": False}
            atomic_json(state_file, state)
        records = (json.loads(catalog_file.read_text(encoding="utf-8"))
                   if catalog_file.is_file() else {"reference": reference})
        if records.get("reference", {}).get("sha256") != reference["sha256"]:
            raise ValueError("The manifest no longer references the audited v1 clean image.")
        images = output / "normal"
        images.mkdir(exist_ok=True)
        for key, record in records.items():
            if key == "reference":
                continue
            path = images / f"{key}.png"
            if key not in views or record["path"] != path.relative_to(ROOT).as_posix():
                raise ValueError(f"Unexpected checkpoint entry {key}")
            check_png(path)
            if sha256(path) != record["sha256"]:
                raise ValueError(f"Checkpoint image changed: {path}")
        if state["complete"]:
            if len(records) != COUNT + 1:
                raise ValueError("The completed manifest is missing normal images.")
            print(f"Already complete: {COUNT} new normal PNGs in {images}", flush=True)
            return

        scene = bpy.context.scene
        gpu = configure_render(scene, config["seed"])
        motion, lights = configure_lighting(scene)
        lights["Rig RGB"] = COLORS["regular"]
        lights.update_tag()
        embedded = bpy.data.texts["generate_tray_anomalies.py"]
        scope = {"__name__": "normals_v2"}
        exec(compile(embedded.as_string(), embedded.name, "exec"), scope)
        defects = scope["generate_anomalies"](seed=config["seed"], counts=dict.fromkeys(scope["COUNT_KEYS"], 0),
                                             scene=scene)
        if defects:
            raise RuntimeError("The normal-only scene contains generated defects.")
        scene.compositing_node_group = None
        scene.render.use_compositing = scene.render.use_sequencer = False
        camera = scene.objects[CAMERA]
        elapsed = []
        print(f"ANNAlyzer regular normals v2 | {gpu} | {SIDE} x {SIDE} | {len(records) - 1}/{COUNT} complete",
              flush=True)
        with TemporaryDirectory(prefix=".render_", dir=output) as temp:
            pending = Path(temp) / "normal.png"
            for key, planned in views.items():
                if key in records:
                    continue
                setup = dict(planned["render_setup"])
                motion["All 4 camera height (m)"] = setup["camera_height_m"]
                motion["All 4 light height (m)"] = setup["led_height_m"]
                motion.update_tag()
                scene.frame_set(1)
                scene.camera = camera
                evaluated = bpy.context.evaluated_depsgraph_get()
                pose = camera.evaluated_get(evaluated).matrix_world.copy()
                led = scene.objects["Front LED vertical slide"].evaluated_get(evaluated).matrix_world.translation.z
                if abs(pose.translation.z - setup["camera_height_m"]) > .0002 or abs(led - setup["led_height_m"]) > .0002:
                    raise ValueError(f"Rig drivers did not reach the requested heights for {key}.")
                print(f"START {len(records)}/{COUNT} | {key} | {planned['split']} | "
                      f"camera {pose.translation.z:.3f} m, LED {led:.3f} m", flush=True)
                start = time.perf_counter()
                render(scene, pending)
                seconds = time.perf_counter() - start
                path = images / f"{key}.png"
                pending.replace(path)
                setup["effective_led_height_m"] = float(led)
                records[key] = {**planned, "render_setup": setup,
                                "camera_matrix_world": [list(row) for row in pose],
                                "focal_length_mm": camera.data.lens, "sensor_width_mm": camera.data.sensor_width,
                                "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                                "render_seconds": round(seconds, 2)}
                atomic_json(catalog_file, records)
                elapsed.append(seconds)
                completed = len(records) - 1
                eta = (COUNT - completed) * sum(elapsed[-12:]) / len(elapsed[-12:])
                print(f"DONE {completed}/{COUNT} | {seconds:.1f}s | ETA {eta / 60:.1f} min", flush=True)
        if len(records) != COUNT + 1:
            raise RuntimeError("The v2 image manifest is incomplete.")
        quality = audit_split(records)
        atomic_json(catalog_file, records)
        atomic_json(state_file, {"configuration": config, "complete": True, "split_audit": quality})
        print(f"COMPLETE | {COUNT} new PNGs + one v1 reference | {output}", flush=True)
        print(f"Cross-split near-duplicate candidates to review: {len(quality['cross_split_candidates'])}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    run(args.output.resolve(), args.resume)
