"""Configuration shared by the Windows launcher, renderer, and experiment runner."""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL = ROOT / ".workstation"
LIGHTS = ("red", "green", "regular")
MODELS = ("patchcore", "anomalydino", "unet", "flownomaly")
DEFAULT = {
    "dataset": "data/workstation_datasets/tray",
    "blender": "",
    "scene": "synthetic_dataset/battery_tray_inspection_anomalies.blend",
    "render": {"pairs": 100, "resolution": 2048, "cycles_samples": 100, "seed": 24117,
               "lights": list(LIGHTS), "gpu_match": "", "camera_height": None, "led_height": None},
    "split": {"train": 70, "validation": 15, "seed": 24117, "accept_near_duplicates": False},
    "training": {"models": list(MODELS), "lights": list(LIGHTS), "seed": 24117, "overlap": 128,
                 "gaussian_sigma": 128.0, "cpu_threads": 4},
    "patchcore": {"candidates": 128, "coreset": 0.1, "query_chunk": 512, "bank_chunk": 8192},
    "anomalydino": {"candidates": 128, "coreset": 0.1, "query_chunk": 512, "bank_chunk": 8192},
    "unet": {"epochs": 10, "batch": 4, "learning_rate": 0.0001, "precision": "fp16"},
    "flownomaly": {"encoder_epochs": 3, "encoder_lr": 0.00003, "flow_epochs": 12, "flow_lr": 0.001,
                   "flow_batch": 512, "tokens_per_tile": 64},
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def rooted(value):
    return (ROOT / value).resolve()


def custom_dataset(value):
    path = rooted(value)
    reserved = (ROOT / "data/annalyzer_batch_2048", ROOT / "data/annalyzer_regular_normals_v2",
                ROOT / "V1", ROOT / "Anamoly_detection", ROOT / "synthetic_dataset", ROOT / ".git")
    if (path == ROOT or ROOT.is_relative_to(path) or path == ROOT / "data"
            or any(path.is_relative_to(p) or p.is_relative_to(path) for p in reserved)):
        raise ValueError(f"Choose a dedicated custom dataset directory: {path}")
    return path


def validate(config):
    def schema(values, defaults, prefix=""):
        if not isinstance(values, dict) or values.keys() != defaults.keys():
            raise ValueError(f"Expected config fields under {prefix or 'root'}: {', '.join(defaults)}")
        for key, example in defaults.items():
            value = values[key]
            field = prefix + key
            if isinstance(example, dict):
                schema(value, example, field + ".")
            elif example is None:
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"{field} must be a positive height in metres or null")
            elif isinstance(example, float):
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{field} must be a finite positive number")
            elif type(value) is not type(example):
                raise ValueError(f"{field} must have type {type(example).__name__}")
            elif isinstance(example, int) and not isinstance(example, bool):
                minimum = 0 if key in ("seed", "overlap") else 1
                if value < minimum:
                    raise ValueError(f"{field} must be at least {minimum}")

    def choices(values, allowed):
        if not values or any(type(value) is not str or value not in allowed for value in values) or len(set(values)) != len(values):
            raise ValueError(f"Select distinct values from {', '.join(allowed)}")

    schema(config, DEFAULT)
    render, split, train = (config[key] for key in ("render", "split", "training"))
    choices(render["lights"], LIGHTS)
    choices(train["lights"], LIGHTS)
    choices(train["models"], MODELS)
    if not config["dataset"].strip() or not config["scene"].strip():
        raise ValueError("Set a dataset folder and Blender scene path.")
    custom_dataset(config["dataset"])
    if render["pairs"] < 3 or render["resolution"] < 512 or train["overlap"] >= 512:
        raise ValueError("Use at least three render pairs, a square resolution >=512, and overlap <512.")
    if max(render["seed"] + 3 * render["pairs"], split["seed"], train["seed"]) > 2_147_483_647:
        raise ValueError("The render or training seed exceeds the supported range.")
    if split["train"] + split["validation"] >= 100:
        raise ValueError("Train and validation percentages must leave a held-out test fold.")
    for name, count in (("patchcore", 4096), ("anomalydino", 1369)):
        if config[name]["coreset"] > 1 or config[name]["candidates"] > count:
            raise ValueError(f"Invalid {name} coreset ratio or tile candidate count")
    if config["flownomaly"]["tokens_per_tile"] > 1369:
        raise ValueError("Flownomaly has 1369 tokens per tile.")
    if config["unet"]["precision"] not in ("fp16", "bf16", "fp32"):
        raise ValueError("U-Net precision must be fp16, bf16 or fp32.")
    return config


def origins(frame, overlap):
    final = frame - 512
    axis = list(range(0, final + 1, 512 - overlap))
    if axis[-1] != final:
        axis.append(final)
    return [(y, x) for y in axis for x in axis]
