"""Render a clean or blistered ANNAlyzer demo image using a fixed camera."""

import argparse
from datetime import datetime
import os
from pathlib import Path
from secrets import randbelow
from struct import pack
import sys
from tempfile import TemporaryDirectory, mkdtemp
from time import perf_counter
from zlib import compress, crc32

import bpy

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_annalyzer_batch import (CAMERA, COLORS, SIDE, THRESHOLD, atomic_json, configure_lighting,
                                    configure_render, render, variant)
from render_prototype_dataset import normal_and_mask_nodes, positive_pixels


OUTPUT = Path(__file__).resolve().parents[1] / "data" / "demo_runs"
KINDS = ("clean", "blistered")
CAMERA_HEIGHT = 1.61
LED_HEIGHT = 1.102
RENDER_SEED = 24117
MAX_SEED = 2**31 - 1


def configure_capture(scene, color, side=SIDE, samples=100, gpu_match="4060",
                      camera_height=CAMERA_HEIGHT, led_height=LED_HEIGHT):
    gpu = configure_render(scene, RENDER_SEED, side=side, samples=samples, gpu_match=gpu_match)
    motion, levels = configure_lighting(scene)
    motion["All 4 camera height (m)"] = camera_height
    motion["Front left camera height (m)"] = 1.575  # Neutral offset on top of the shared height.
    motion["All 4 light height (m)"] = led_height
    levels["Rig RGB"] = COLORS[color]
    if color != "regular":
        levels["Rig LEDs"], levels["Ceiling LEDs"] = 1.75, .75
    motion.update_tag()
    levels.update_tag()
    scene.render.film_transparent = False
    scene.render.use_motion_blur = False
    scene.frame_set(1)
    camera = scene.objects[CAMERA]
    scene.camera = camera
    depsgraph = bpy.context.evaluated_depsgraph_get()
    pose = camera.evaluated_get(depsgraph).matrix_world.copy()
    led = scene.objects["Front LED vertical slide"].evaluated_get(depsgraph).matrix_world.translation.z
    if abs(pose.translation.z - camera_height) > .0002 or abs(led - led_height) > .0002:
        raise ValueError("The saved scene did not reach the demo camera or LED height.")
    return {
        "camera": CAMERA, "camera_matrix_world": [list(row) for row in pose],
        "focal_length_mm": camera.data.lens, "sensor_width_mm": camera.data.sensor_width,
        "camera_height_m": pose.translation.z, "led_height_m": led,
        "rig_rgb": list(COLORS[color]), "rig_intensity": levels["Rig LEDs"],
        "ceiling_intensity": levels["Ceiling LEDs"], "dark_field_mix": motion["Dark-field mix"],
        "exposure_ev": scene.view_settings.exposure, "frame": 1, "device": gpu,
    }


def write_black_mask(path, side=SIDE):
    """Write an 8-bit grayscale PNG with zero foreground pixels."""
    def chunk(kind, data):
        return pack(">I", len(data)) + kind + data + pack(">I", crc32(kind + data))

    header = pack(">IIBBBBB", side, side, 8, 0, 0, 0, 0)
    rows = bytes(side * (side + 1))  # Each row begins with a zero PNG filter byte.
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) +
                     chunk(b"IDAT", compress(rows)) + chunk(b"IEND", b""))


def run(color, kind, seed, output_root, *, side=SIDE, samples=100, gpu_match="4060",
        camera_height=CAMERA_HEIGHT, led_height=LED_HEIGHT):
    if not bpy.app.background or not bpy.data.filepath:
        raise RuntimeError("Run the demo with the saved inspection scene in background Blender.")
    scene = bpy.context.scene
    capture = configure_capture(scene, color, side, samples, gpu_match, camera_height, led_height)
    source = bpy.data.texts["generate_tray_anomalies.py"]
    scope = {"__name__": "demo_anomalies"}
    exec(compile(source.as_string(), source.name, "exec"), scope)
    generate = scope["generate_anomalies"]
    controls = scene.objects[scope["CONTROL"]]
    controls["Enabled"] = controls["Blister amount"] = 1.
    clean_counts = dict.fromkeys(scope["COUNT_KEYS"], 0)

    output_root.mkdir(parents=True, exist_ok=True)
    blistered = kind == "blistered"
    seed_suffix = f"_{seed}" if blistered else ""
    prefix = f"{datetime.now():%Y%m%d_%H%M%S}_{color}_{kind}{seed_suffix}_"
    output = Path(mkdtemp(prefix=prefix, dir=output_root))
    metadata = {
        "complete": False, "lighting": color, "kind": kind, "blister_seed": seed if blistered else None,
        "cycles_seed": RENDER_SEED, "scene": Path(bpy.data.filepath).name, "capture": capture,
        "resolution": [side, side], "cycles_samples": scene.cycles.samples, "denoising": True,
        "noise_threshold": THRESHOLD, "images": {},
    }
    manifest = output / "demo.json"
    atomic_json(manifest, metadata)
    print(f"OUTPUT {output}", flush=True)
    print(f"SETUP {color} | {kind} | {side} x {side} | {capture['device']}", flush=True)
    if blistered:
        print(f"BLISTER SEED {seed}", flush=True)

    with TemporaryDirectory(prefix=".render_", dir=output) as temp:
        scratch = Path(temp)
        counts, settings = variant(seed, clean_counts) if blistered else (clean_counts, {})
        for key, value in settings.items():
            controls[key] = value
        defects = generate(seed=seed if blistered else RENDER_SEED, counts=counts, scene=scene)
        scene.camera = scene.objects[CAMERA]
        if blistered:
            mask_node = normal_and_mask_nodes(scene, scratch)
            mask_node.file_name = "demo_mask"
        else:
            if defects:
                raise ValueError("The clean scene contains generated defects.")
            scene.render.use_compositing = scene.render.use_sequencer = False
        temporary_image = scratch / f"{kind}.png"
        print(f"START 1/1 | {kind}", flush=True)
        start = perf_counter()
        render(scene, temporary_image, side)
        result = {"image": temporary_image.name, "render_seconds": round(perf_counter() - start, 2),
                  "mask": "blister_mask.png", "mask_positive_pixels": 0}
        if blistered:
            files = list(scratch.glob("demo_mask*.png"))
            if len(files) != 1:
                raise ValueError(f"Expected one blister mask and found {len(files)}.")
            mask = files[0]
            result.update(mask_positive_pixels=positive_pixels(mask, side), flange_blisters=counts["flange_blisters"],
                          cross_ply_blisters=counts["crossply_blisters"], size_settings=settings, instances=defects)
        else:
            mask = scratch / "black_mask.png"
            write_black_mask(mask, side)
        mask.replace(output / result["mask"])
        temporary_image.replace(output / temporary_image.name)
        metadata["images"][kind] = result
        print(f"DONE 1/1 | {kind} | {result['render_seconds']:.1f} s | mask saved", flush=True)
    metadata["complete"] = True
    atomic_json(manifest, metadata)
    print(f"COMPLETE | {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lighting", required=True, choices=COLORS)
    parser.add_argument("--kind", required=True, choices=KINDS)
    parser.add_argument("--seed", help="Omit or leave blank for a random blister seed.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--resolution", type=int, default=SIDE)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--gpu-match", default="4060")
    parser.add_argument("--camera-height", type=float, default=CAMERA_HEIGHT)
    parser.add_argument("--led-height", type=float, default=LED_HEIGHT)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    typed_seed = args.seed if args.seed is not None else os.getenv("ANNALYZER_DEMO_SEED", "")
    try:
        seed = int(typed_seed) if typed_seed.strip() else randbelow(MAX_SEED + 1)
        if not 0 <= seed <= MAX_SEED:
            raise ValueError
    except ValueError:
        parser.error(f"Seed must be a whole number from 0 to {MAX_SEED} or left blank.")
    if args.resolution < 512 or args.samples < 1:
        parser.error("Use a resolution of at least 512 pixels and one or more Cycles samples.")
    run(args.lighting, args.kind, seed, args.output_root.resolve(), side=args.resolution,
        samples=args.samples, gpu_match=args.gpu_match, camera_height=args.camera_height,
        led_height=args.led_height)
