"""Local V3 inference over a newline-delimited JSON pipe."""

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

ANOMALY = Path(__file__).resolve().parent.parent
FRAME = 2048
TILE = 512
OVERLAP = 128
STRIDE = TILE - OVERLAP
ORIGINS = [(x, y) for y in range(0, FRAME - TILE + 1, STRIDE) for x in range(0, FRAME - TILE + 1, STRIDE)]
LIGHTS = ("red", "green", "regular")
MODELS = {"patchcore": "PatchCore", "anomalydino": "AnomalyDINO", "unet": "U-Net"}


def frame_layout(width, height):
    scale = FRAME / max(width, height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    return (FRAME - w) // 2, (FRAME - h) // 2, w, h


def catalog_entry(model, light):
    results = ANOMALY / model / "results" / "V3"
    for marker in sorted(results.glob("all_lights_v3_*/complete.json"), reverse=True):
        run = marker.parent
        validation = run / light / "validation" / "report.json"
        if not validation.is_file():
            continue
        report = json.loads(validation.read_text(encoding="utf-8"))
        if (report.get("model"), report.get("lighting"), report.get("stage")) != (MODELS[model], light, "validation"):
            raise ValueError(f"Unexpected model, lighting or stage in {validation}")
        checkpoint = (run / report["checkpoint"]["file"]).resolve()
        if not checkpoint.is_relative_to(run.resolve()) or not checkpoint.is_file():
            raise FileNotFoundError(f"Missing V3 checkpoint referenced by {validation}: {checkpoint}")
        return report, checkpoint, run.name
    raise FileNotFoundError(f"No completed V3 {MODELS[model]} run with a {light} validation report under {results}")


def load_model(name, light, progress):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    sys.path.insert(0, str(ANOMALY))
    from v2_workflow import cuda_environment, sha
    from v3_workflow import load_module

    progress("Finding the saved validation run…")
    report, checkpoint, run = catalog_entry(name, light)
    progress("Verifying the fitted checkpoint…")
    if sha(checkpoint) != report["checkpoint"]["sha256"]:
        raise ValueError(f"Checkpoint does not match the saved validation report: {checkpoint}")
    folder = ANOMALY / name
    progress("Initializing CUDA and loading the model…")
    if name != "unet":
        runner = load_module(folder / "V2/run_v2.py", f"{name}_inspection")
        infer = runner.build(checkpoint, train=None, defects=None).infer
    else:
        sys.path.insert(0, str(ANOMALY / "patchcore/V2"))
        trainer = load_module(folder / "V1/run.py", "unet_inspection")
        torch = trainer.torch
        cuda_environment(trainer.SEED, fp32=False)
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        model = trainer.build_model().eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"])
        mean, std = (torch.tensor(values, device="cuda").view(1, 3, 1, 1)
                     for values in (trainer.RGB_MEAN, trainer.RGB_STD))
        stitcher = trainer.GaussianStitcher()

        def infer(rgb):
            return stitcher.infer(model, rgb, mean, std)

    return infer, report, run


def read_tray(image):
    import numpy as np
    from PIL import Image

    with Image.open(image) as source:
        if source.format != "PNG" or source.mode not in ("RGB", "RGBA"):
            raise ValueError("Select an RGB Blender PNG.")
        if source.mode == "RGBA" and source.getchannel("A").getextrema() != (255, 255):
            raise ValueError("Render with an opaque background.")
        return np.array(source.convert("RGB"))


def inspect(infer, report, image, overlay, progress, heatmap=None):
    import numpy as np
    from matplotlib import colormaps
    from PIL import Image

    progress("Reading the tray image…")
    rgb = read_tray(image)
    height, width = rgb.shape[:2]
    model_input = rgb
    if (height, width) != (FRAME, FRAME):
        progress("Fitting the image to the model input…")
        x, y, w, h = frame_layout(width, height)
        method = Image.Resampling.LANCZOS if max(height, width) > FRAME else Image.Resampling.BILINEAR
        resized = np.asarray(Image.fromarray(rgb).resize((w, h), method))
        model_input = np.pad(resized, ((y, FRAME - h - y), (x, FRAME - w - x), (0, 0)), mode="edge")
    progress(f"Running inference on {len(ORIGINS)} overlapping tiles…")
    start = perf_counter()
    score, image_score = infer(model_input)
    seconds = perf_counter() - start
    if score.shape != (FRAME, FRAME) or not np.isfinite(score).all() or not math.isfinite(image_score):
        raise ValueError("The model returned an invalid score or map.")
    display_max = float(report["display_max"])
    if not math.isfinite(display_max) or display_max <= 0:
        raise ValueError("Invalid heatmap scale in the validation report.")
    progress("Preparing the prediction views…")
    if (height, width) != (FRAME, FRAME):
        score = np.asarray(Image.fromarray(score[y:y + h, x:x + w]).resize((width, height), Image.Resampling.BILINEAR))
    intensity = np.clip(score / display_max, 0.0, 1.0)
    heat = colormaps["magma"](intensity, bytes=True)[..., :3]
    if heatmap is not None:
        Image.fromarray(heat).save(heatmap)
    strength = (0.75 * intensity)[..., None]
    Image.fromarray((rgb * (1.0 - strength) + heat * strength).astype(np.uint8)).save(overlay)
    threshold = report.get("threshold")
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError("Invalid threshold in the validation report.")
    return score, {"score": float(image_score), "threshold": threshold, "display_max": display_max,
                   "flagged": None if threshold is None else bool(image_score > threshold), "seconds": seconds}


def read_mask(path, size):
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        if source.format != "PNG" or source.size != size:
            raise ValueError(f"Select a ground-truth PNG matching the tray image ({size[0]} × {size[1]}).")
        if source.mode in ("1", "L") and "transparency" not in source.info:
            values = np.asarray(source.convert("L"))
        else:
            if source.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
                raise ValueError("Use an 8-bit black-and-white ground-truth PNG.")
            pixels = np.asarray(source.convert("RGBA"))
            values = pixels[..., 0]
            if not (pixels[..., 3] == 255).all() or not (pixels[..., :3] == values[..., None]).all():
                raise ValueError("The mask must be opaque, with white defects on a black background.")
        if not ((values == 0) | (values == 255)).all():
            raise ValueError("The mask must be binary, with white defects on a black background.")
        return values == 255


def compare_mask(score, flagged, image, mask, overlay, progress):
    import numpy as np
    from PIL import Image
    from sklearn.metrics import average_precision_score, roc_auc_score

    progress("Checking the ground-truth mask…")
    foreground = read_mask(mask, (score.shape[1], score.shape[0]))
    defect_pixels = int(np.count_nonzero(foreground))
    agreement = None if flagged is None else flagged == bool(defect_pixels)
    ap = auroc = None
    if 0 < defect_pixels < foreground.size:
        progress("Calculating pixel AP and AUROC…")
        labels, scores = foreground.ravel(), score.ravel()
        ap = float(average_precision_score(labels, scores))
        auroc = float(roc_auc_score(labels, scores))
    progress("Preparing the ground-truth overlay…")
    rgb = read_tray(image)
    rgb[foreground] = 0.35 * rgb[foreground] + 0.65 * np.array((48, 235, 133))
    Image.fromarray(rgb).save(overlay)
    return {"defect_pixels": defect_pixels, "agreement": agreement, "pixel_ap": ap, "pixel_auroc": auroc}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("lighting", choices=LIGHTS)
    args = parser.parse_args()
    wire = sys.stdout

    def send(event, **fields):
        print(json.dumps({"event": event, **fields}, allow_nan=False), file=wire, flush=True)

    def progress(message):
        send("progress", message=message)

    with redirect_stdout(sys.stderr):
        try:
            infer, report, run = load_model(args.model, args.lighting, progress)
        except Exception as exc:
            traceback.print_exc()
            send("error", message=str(exc))
            return 1
        send("ready", run=run, protocol=report["configuration"]["protocol"])
        prediction = None
        for line in sys.stdin:
            try:
                request = json.loads(line)
                action = request.get("action", "inspect")
                if action == "inspect":
                    prediction = None
                    score, result = inspect(infer, report, request["image"], request["overlay"], progress,
                                            heatmap=request.get("heatmap"))
                    prediction = (score, result["flagged"], request["image"])
                    send("result", **result)
                elif action == "compare":
                    if prediction is None:
                        raise ValueError("Inspect the tray before adding a ground-truth mask.")
                    comparison = compare_mask(*prediction, request["mask"], request["overlay"], progress)
                    send("comparison", **comparison)
                else:
                    raise ValueError("Unknown inspection request.")
            except Exception as exc:
                traceback.print_exc()
                send("error", message=str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
