"""Render the regular, red and green ANNAlyzer datasets in background Blender."""

import argparse
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import msvcrt
from pathlib import Path
import random
import shutil
import sys
from tempfile import TemporaryDirectory
import time

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_prototype_dataset import CAMERA, SIDE, normal_and_mask_nodes, png_header, positive_pixels, render


PER_SET = 100
PAIR_COUNT = 3 * PER_SET
RENDER_COUNT = 1 + 5 * PER_SET
SAMPLES = 100
THRESHOLD = .015
COLORS = {
    "regular": (1., .94, .88),
    "red": (1., 0., 0.),
    "green": (0., 1., 0.),
}
BANDS = (
    ("low", 34, (1.28, 1.39), (1.102, 1.26)),
    ("mid", 33, (1.39, 1.50), (1.26, 1.43)),
    ("high", 33, (1.50, 1.61), (1.43, 1.60)),
)
RIG_INTENSITY = (1.50, 2.00)
CEILING_INTENSITY = (.35, 1.10)
DARK_FIELD_MIX = (.65, .90)
DARK_FIELD_COUNTS = {"low": 9, "mid": 8, "high": 8}
GRAZING_HEIGHT = 1.065
GRAZING_CEILING_FRACTION = .025
DIRECTORIES = ("normal", "bubbles", "masks")
SAMPLE_IDS = tuple(f"{index:04}" for index in range(1, PER_SET + 1))


@dataclass(frozen=True)
class Setup:
    band: str
    camera_height: float
    led_height: float
    rig_intensity: float = 1.
    ceiling_intensity: float = 1.
    dark_field_mix: float = 0.


def varied_setups(seed):
    rng = random.Random(seed ^ 0x61A099)

    def stratified(bounds, count):
        lower, upper = bounds
        step = (upper - lower) / count
        values = [round(lower + step * (i + rng.uniform(.15, .85)), 6) for i in range(count)]
        rng.shuffle(values)
        return values

    schedule = []
    for band, count, camera_range, led_range in BANDS:
        dark_count = DARK_FIELD_COUNTS[band]
        dark_mix = dict(zip(rng.sample(range(count), dark_count), stratified(DARK_FIELD_MIX, dark_count)))
        settings = [stratified(bounds, count)
                    for bounds in (camera_range, led_range, RIG_INTENSITY, CEILING_INTENSITY)]
        schedule.extend(Setup(band, *values, dark_mix.get(index, 0.))
                        for index, values in enumerate(zip(*settings)))
    assert len(schedule) == PER_SET and len(set(schedule)) == PER_SET
    return schedule


def atomic_json(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pending.replace(path)


@contextmanager
def output_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".batch.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(f"Another batch already has this output folder open: {folder}") from exc
        yield


def configure_render(scene, base_seed):
    scene.frame_set(1)
    scene.camera = scene.objects[CAMERA]
    render_settings = scene.render
    render_settings.engine = "CYCLES"
    scene.cycles.device = "GPU"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "OPTIX"
    preferences.refresh_devices()
    for device in preferences.devices:
        if device.type in ("CPU", "OPTIX"):
            device.use = device.type == "OPTIX" and "4060" in device.name
    gpu = next((d.name for d in preferences.devices if d.type == "OPTIX" and d.use), None)
    if gpu is None:
        raise RuntimeError("The RTX 4060 OptiX device used for the timing pilot was not found.")
    scene.cycles.samples = SAMPLES
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = THRESHOLD
    scene.cycles.use_denoising = True
    scene.cycles.seed = base_seed
    scene.cycles.use_animated_seed = False
    render_settings.use_persistent_data = True
    render_settings.resolution_x = render_settings.resolution_y = SIDE
    render_settings.resolution_percentage = 100
    render_settings.pixel_aspect_x = render_settings.pixel_aspect_y = 1.
    render_settings.use_border = False
    render_settings.use_multiview = False
    render_settings.use_file_extension = True
    image = render_settings.image_settings
    image.file_format = "PNG"
    image.color_mode = "RGB"
    image.color_depth = "8"
    image.color_management = "FOLLOW_SCENE"
    scene.view_settings.exposure = 1.
    scene.view_settings.use_white_balance = False
    return gpu


def configure_lighting(scene):
    motion = scene.objects["Capture controls"]
    lights = scene.objects["Lighting levels · select for custom properties"]
    lights["Ceiling LEDs"] = lights["Rig LEDs"] = 1.
    lights["Ceiling RGB"] = (1., .91, .80)
    lights["World fill strength"] = 0.
    for group in ("Rig", "Ceiling"):
        lights[f"{group} Use Kelvin"] = False
        lights[f"{group} beam spread (deg)"] = 180.
        lights[f"{group} emitter size scale"] = 1.
        lights[f"{group} Shadows"] = True
        for response in ("Diffuse", "Specular", "Transmission", "Volume"):
            lights[f"{group} {response} response"] = 1.
    motion["Dark-field mix"] = 0.
    motion["Grazing height (m)"] = GRAZING_HEIGHT
    motion["Grazing down angle (deg)"] = 9.
    motion["Grazing beam width (deg)"] = 28.
    motion["Ceiling fraction in dark field"] = GRAZING_CEILING_FRACTION
    for bar in ("Front", "Rear", "Left end", "Right end"):
        motion[f"{bar} bar intensity"] = 1.
        motion[f"{bar} bar height (m)"] = 1.602
        motion[f"{bar} bar down angle (deg)"] = 71.
        lights[f"{bar} bar RGB tint"] = (1., 1., 1.)
        lights[f"{bar} bar Override Kelvin"] = False
        lights[f"{bar} bar Override Beam"] = False
    return motion, lights


def variant(seed, clean_counts):
    rng = random.Random(seed ^ 0x719583)
    counts = clean_counts | {"flange_blisters": rng.randint(7, 10),
                             "crossply_blisters": rng.randint(2, 8)}
    settings = {}
    for region, key in (("Flange", "flange_blisters"), ("Cross-ply", "crossply_blisters")):
        settings[f"{region} blister width scale"] = 1.
        settings[f"{region} blister lift scale"] = rng.uniform(.82, 1.26)
        for number in range(1, counts[key] + 1):
            settings[f"{region} blister {number:02d} width"] = rng.uniform(.72, 1.24)
    return counts, settings


def check_png(path, grayscale=False):
    expected = (SIDE, SIDE, 8, 0 if grayscale else 2)
    if png_header(path) != expected:
        raise ValueError(f"Unexpected PNG dimensions or encoding: {path}")


def open_catalog(folder, contract, resume):
    configuration = folder / "dataset.json"
    catalog_file = folder / "samples.json"
    if resume:
        prior = json.loads(configuration.read_text(encoding="utf-8"))
        if prior["configuration"] != contract:
            raise ValueError("Resume settings differ. Use the original scene file, script and seed.")
        records = json.loads(catalog_file.read_text(encoding="utf-8")) if catalog_file.exists() else {}
    else:
        if any(p.name != ".batch.lock" for p in folder.iterdir()):
            raise FileExistsError("Output folder is not empty. Pick a new folder or use --resume.")
        atomic_json(configuration, {"configuration": contract, "complete": False})
        records = {}
    for color in COLORS:
        for kind in DIRECTORIES:
            (folder / color / kind).mkdir(parents=True, exist_ok=True)
    for key, record in records.items():
        color, sample = key.split("/")
        if color not in COLORS or sample not in SAMPLE_IDS:
            raise ValueError(f"Unexpected sample key: {key}")
        for kind in DIRECTORIES:
            check_png(folder / color / kind / (sample + ".png"), grayscale=kind == "masks")
        if record["flange_count"] < 7 or record["mask_positive_pixels"] <= 0:
            raise ValueError(f"Invalid completed sample: {key}")
    return records


def run(folder, seed, resume):
    started = time.perf_counter()
    scene = bpy.context.scene
    blend = Path(bpy.data.filepath)
    if not blend.is_file() or not bpy.app.background:
        raise RuntimeError("Open the saved inspection cell in background Blender to run this batch.")
    source = bpy.data.texts["generate_tray_anomalies.py"]
    script = Path(__file__).resolve()
    helper = script.with_name("render_prototype_dataset.py")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (blend, script, helper)}
    contract = {
        "source_sha256": source_hashes,
        "seed": seed, "camera": CAMERA, "resolution": [SIDE, SIDE],
        "cycles_samples": SAMPLES, "noise_threshold": THRESHOLD,
        "denoising": True, "device": "RTX 4060 OptiX", "encoding": "8-bit PNG",
        "rig_rgb": {name: list(rgb) for name, rgb in COLORS.items()},
        "varied_height_split": {"low": 34, "mid": 33, "high": 33},
        "colored_set_lighting": {
            "rig_intensity_range": list(RIG_INTENSITY),
            "ceiling_base_intensity_range": list(CEILING_INTENSITY),
            "dark_field_mix_range": list(DARK_FIELD_MIX),
            "dark_field_examples_by_band_per_color": DARK_FIELD_COUNTS,
            "full_dark_field_led_height_m": GRAZING_HEIGHT,
            "full_dark_field_ceiling_fraction": GRAZING_CEILING_FRACTION,
        },
        "regular_heights_m": {"camera": 1.61, "led": 1.102},
        "expected_pairs": PAIR_COUNT, "expected_renders": RENDER_COUNT,
        "normal_regular": "Copies of one reference because all scene and sampling settings are fixed.",
        "mask": "Grayscale 0 for background and 255 for visible blister pixels.",
    }
    records = open_catalog(folder, contract, resume)
    if len(records) == PAIR_COUNT:
        configuration = folder / "dataset.json"
        if not json.loads(configuration.read_text(encoding="utf-8"))["complete"]:
            atomic_json(configuration, {"configuration": contract, "complete": True})
        print(f"Already complete: {PAIR_COUNT} matched sets in {folder}", flush=True)
        return

    gpu = configure_render(scene, seed)
    motion, lights = configure_lighting(scene)
    scope = {"__name__": "batch_anomalies"}
    exec(compile(source.as_string(), source.name, "exec"), scope)
    generate = scope["generate_anomalies"]
    controls = scene.objects[scope["CONTROL"]]
    controls["Enabled"] = controls["Blister amount"] = 1.
    clean_counts = dict.fromkeys(scope["COUNT_KEYS"], 0)
    generate(seed=seed, counts=clean_counts, scene=scene)

    schedule = varied_setups(seed)
    reference = folder / "regular" / "normal" / "0001.png"
    rendered = len(records) + int(reference.is_file())
    rendered += sum((folder / color / "normal" / (sample + ".png")).is_file()
                    for color in ("red", "green") for sample in SAMPLE_IDS)
    recent_timings = deque(maxlen=12)
    print(f"ANNAlyzer | {gpu} | {SIDE} × {SIDE} | {SAMPLES} samples | {folder}", flush=True)
    print(f"PROGRESS | {len(records)}/{PAIR_COUNT} pairs | {rendered}/{RENDER_COUNT} renders", flush=True)

    with TemporaryDirectory(prefix=".render_", dir=folder) as scratch_directory:
        scratch = Path(scratch_directory)
        save_mask = normal_and_mask_nodes(scene, scratch)
        staged = scratch / "image.png"

        def capture(sample, kind, destination):
            nonlocal rendered
            print(f"START render {rendered + 1}/{RENDER_COUNT} | {sample} | {kind}", flush=True)
            tick = time.perf_counter()
            scene.camera = scene.objects[CAMERA]
            render(scene, staged)
            seconds = time.perf_counter() - tick
            staged.replace(destination)
            rendered += 1
            recent_timings.append(seconds)
            remaining = timedelta(seconds=round((RENDER_COUNT - rendered) *
                                                sum(recent_timings) / len(recent_timings)))
            print(f"DONE  render {rendered}/{RENDER_COUNT} | {seconds:.1f}s | "
                  f"{len(records)}/{PAIR_COUNT} pairs | render ETA {remaining}", flush=True)
            return round(seconds, 2)

        for color_index, (color, rgb) in enumerate(COLORS.items()):
            setups = [Setup("fixed", 1.61, 1.102)] * PER_SET if color == "regular" else schedule
            lights["Rig RGB"] = rgb
            for index, setup in enumerate(setups, 1):
                name = f"{index:04}"
                key = f"{color}/{name}"
                if key in records:
                    continue
                motion["All 4 camera height (m)"] = setup.camera_height
                motion["All 4 light height (m)"] = setup.led_height
                motion["Dark-field mix"] = setup.dark_field_mix
                lights["Rig LEDs"] = setup.rig_intensity
                lights["Ceiling LEDs"] = setup.ceiling_intensity
                motion.update_tag()
                lights.update_tag()
                print(f"SETUP {key} | {setup.band} | rig {setup.rig_intensity:.2f}× | "
                      f"ceiling {setup.ceiling_intensity:.2f}× base | "
                      f"dark-field mix {setup.dark_field_mix:.2f}", flush=True)
                paths = {kind: folder / color / kind / (name + ".png") for kind in DIRECTORIES}
                normal_path = paths["normal"]
                clean_seconds = None
                clean_target = reference if color == "regular" else normal_path
                if not clean_target.is_file():
                    generate(seed=seed, counts=clean_counts, scene=scene)
                    save_mask.mute = True
                    clean_seconds = capture(key, "clean", clean_target)
                else:
                    check_png(clean_target)
                if normal_path != clean_target:
                    shutil.copyfile(clean_target, staged)
                    staged.replace(normal_path)

                anomaly_seed = seed + color_index * PER_SET + index
                counts, size_settings = variant(anomaly_seed, clean_counts)
                for setting, value in size_settings.items():
                    controls[setting] = value
                defects = generate(seed=anomaly_seed, counts=counts, scene=scene)
                assert sum(d["surface"] == "flange" and d["class"] == "blister"
                           for d in defects) == counts["flange_blisters"]
                save_mask.file_name = "blister_mask"
                save_mask.mute = False
                blister_seconds = capture(key, "blisters + mask", paths["bubbles"])
                masks = list(scratch.glob("blister_mask*.png"))
                if len(masks) != 1:
                    raise RuntimeError(f"Expected one compositor mask for {key}, found {masks}")
                pixels = positive_pixels(masks[0])
                masks[0].replace(paths["masks"])
                evaluated = bpy.context.evaluated_depsgraph_get()
                slide = scene.objects["Front LED vertical slide"].evaluated_get(evaluated)
                records[key] = {
                    "rig_color": color, "height_band": setup.band,
                    "camera_height_m": setup.camera_height, "led_height_m": setup.led_height,
                    "effective_led_height_m": float(slide.matrix_world.translation.z),
                    "rig_intensity": setup.rig_intensity, "ceiling_base_intensity": setup.ceiling_intensity,
                    "dark_field_mix": setup.dark_field_mix,
                    "seed": anomaly_seed, "flange_count": counts["flange_blisters"],
                    "cross_ply_count": counts["crossply_blisters"], "size_settings": size_settings,
                    "mask_positive_pixels": pixels, "normal_render_seconds": clean_seconds,
                    "defect_render_seconds": blister_seconds, "instances": defects,
                    "files": {kind: path.relative_to(folder).as_posix() for kind, path in paths.items()},
                }
                atomic_json(folder / "samples.json", records)
                print(f"PAIR {len(records)}/{PAIR_COUNT} ({100 * len(records) / PAIR_COUNT:.1f}%) | "
                      f"{key} | {setup.band} | seed {anomaly_seed} | "
                      f"{counts['flange_blisters']} flange, {counts['crossply_blisters']} cross-ply", flush=True)

    assert len(records) == PAIR_COUNT and rendered == RENDER_COUNT
    elapsed = round(time.perf_counter() - started, 2)
    atomic_json(folder / "dataset.json", {"configuration": contract, "complete": True,
                                         "last_session_elapsed_seconds": elapsed})
    print(f"COMPLETE | {PAIR_COUNT} matched pairs | {3 * PAIR_COUNT} PNGs | {folder}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Dataset folder")
    parser.add_argument("--seed", type=int, default=24117)
    parser.add_argument("--resume", action="store_true", help="Continue in an existing batch folder")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not 0 <= args.seed <= 2147483647 - PAIR_COUNT:
        parser.error("--seed is outside the supported range")
    output_folder = args.output.resolve()
    with output_lock(output_folder):
        run(output_folder, args.seed, args.resume)
