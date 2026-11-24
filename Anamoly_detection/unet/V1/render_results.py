"""Export figures and combine the three models' held-out validation results."""

import argparse
import csv
import json
from pathlib import Path

import run
import numpy as np
from matplotlib import colormaps, pyplot as plt
from PIL import Image, ImageDraw
from phase3_metrics import SIZE_LABELS


MODELS = ("PatchCore", "AnomalyDINO", "U-Net")
SHADES = ("#4a628b", "#ad663c", "#148073")
FIELDS = ("pixel_auroc", "pixel_ap", "aupro", "image_auroc", "image_ap")
TITLES = ("Pixel AUROC", "Pixel AP", "AUPRO at 30% FPR", "Image AUROC", "Image AP")
LIGHTS = ("Regular", "Red", "Green")


def annotated_figures(out, unet, audit):
    colorize = colormaps["inferno"]
    sheet = Image.new("RGB", (1656, 928), "#191919")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), "U-Net validation | shared probability scale 0 to 1 | Regular uses red-to-regular transfer", fill="white")
    for row_index, colour in enumerate(run.COLOURS):
        folder = out / colour / "visuals"
        folder.mkdir(exist_ok=True)
        previews = ["0004", "0074", "0097"] if colour == "regular" else ["0008", "0041", "0042"]
        for prediction in unet[colour]["predictions"]:
            if prediction["role"] != "bubbles":
                continue
            key = prediction["sample_id"]
            number = key.split("/")[-1]
            assets = audit[key]["files"]
            rgb = run.load_image(assets["bubbles"], "RGB")
            mask = run.load_image(assets["masks"], "L")
            probability = np.load(folder.parent / prediction["probability_path"], allow_pickle=False)
            run.require(probability.shape == (run.FRAME, run.FRAME) and probability.dtype == np.float32
                        and np.isfinite(probability).all() and 0 <= probability.min() <= probability.max() <= 1,
                        f"Invalid probability map {key}")
            mask_pixels = mask == 255
            truth = rgb.copy()
            truth[mask_pixels] = np.round(0.45 * truth[mask_pixels] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
            heat = colorize(probability, bytes=True)[..., :3]
            alpha = (0.65 * probability)[..., None]
            blended = np.rint(rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
            Image.fromarray(mask).save(folder / f"{number}_mask.png")
            Image.fromarray(truth).save(folder / f"{number}_ground_truth_overlay.png")
            Image.fromarray(heat).save(folder / f"{number}_probability.png")
            Image.fromarray(blended).save(folder / f"{number}_prediction_overlay.png")
            if number in previews:
                index = previews.index(number)
                y = 35 + row_index * 294
                for offset, (label, array) in enumerate((("ground truth", truth), ("probability", heat))):
                    x = 8 + (2 * index + offset) * 275
                    draw.text((x, y), f"{key} · {label}", fill="white")
                    sheet.paste(Image.fromarray(array).resize((265, 265), Image.Resampling.LANCZOS), (x, y + 18))
    sheet.save(out / "lighting_predictions.png")


def loss_curves(out, unet):
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    for colour, shade in (("red", "#ac4745"), ("green", "#27866a")):
        history = unet[colour]["history"]
        epochs = [h["epoch"] for h in history]
        for field, style in (("bce_plus_dice", "-"), ("bce", "--"), ("dice_loss", ":")):
            axes[0].plot(epochs, [h[field] for h in history], style, color=shade,
                         label=f"{colour} {field.replace('_', ' ')}")
        axes[1].plot(epochs, [h["validation_pixel_ap"] for h in history], "o-", color=shade, label=colour)
        winner = unet[colour]["best_epoch"]
        axes[1].scatter(winner, history[winner - 1]["validation_pixel_ap"], marker="*", s=105, color=shade, zorder=5)
    for axis, title in zip(axes, ("Training loss", "Validation pixel AP"), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, frameon=False)
    figure.savefig(out / "training_curves.png", dpi=180)
    plt.close(figure)


def write_table(out, models):
    with (out / "three_model_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("model", "validation_lighting", "training_protocol", "normal_training", "defective_training",
                         "validation_normal", "validation_defective", *FIELDS, "mean_inference_s"))
        for model, rows in zip(MODELS, models, strict=True):
            for colour in run.COLOURS:
                row = rows[colour]
                if model == "U-Net":
                    protocol = row["protocol"]
                    fitting = rows["red"] if colour == "regular" else row
                    counts = (fitting["training_normal_count"], fitting["training_defect_count"])
                else:
                    protocol = "fixed regular normal reference" if colour == "regular" else "within-light normal-only"
                    counts = (len(row["train_ids"]), 0)
                metrics = row["metrics"]
                writer.writerow((model, colour, protocol, *counts, 0 if colour == "regular" else 15,
                                 len(row["validation_ids"]), *(metrics[k] if metrics[k] is not None else "N/A" for k in FIELDS),
                                 metrics["mean_inference_s"]))


def metric_charts(out, models):
    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True, layout="constrained")
    for axis, field, title in zip(axes.flat, FIELDS, TITLES):
        for index, (label, rows, shade) in enumerate(zip(MODELS, models, SHADES, strict=True)):
            for slot, colour in enumerate(run.COLOURS):
                xpos = slot + (index - 1) * 0.24
                value = rows[colour]["metrics"][field]
                if value is None:
                    axis.text(xpos, 0.04, "N/A", fontsize=7, ha="center", color=shade)
                else:
                    axis.bar(xpos, value, width=0.22, color=shade, label=label if slot == 1 else None)
                    axis.text(xpos, value + 0.025, f"{value:.2f}", fontsize=7, ha="center")
        axis.set_xlim(-0.55, 2.55)
        axis.set_ylim(0, 1.1)
        axis.set_xticks(range(3), LIGHTS)
        axis.tick_params(labelbottom=True)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.03, 0.65, "Regular U-Net uses the red-trained checkpoint.\nImage metrics are N/A for regular because\nits validation set has zero normal frames.",
                       transform=axes.flat[-1].transAxes, fontsize=10)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", frameon=False, ncols=3)
    figure.savefig(out / "three_model_comparison.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True, layout="constrained")
    for i, (field, title) in enumerate((("aupro", "AUPRO at 30% FPR"), ("mean_overlap_at_1pct_fpr", "Mean overlap at 1% FPR"))):
        for j, (colour, name) in enumerate(zip(run.COLOURS, LIGHTS, strict=True)):
            axis = axes[i, j]
            for index, (model, rows, shade) in enumerate(zip(MODELS, models, SHADES, strict=True)):
                measures = rows[colour]["metrics"]["size_specific"]
                axis.bar(np.arange(3) + (index - 1) * 0.24, [measures[k][field] for k in SIZE_LABELS],
                         width=0.22, color=shade, label=model)
            axis.set_xticks(range(3), ("<64", "64–511", "≥512"))
            axis.set_title(name + (" (U-Net transfer)" if colour == "regular" else ""))
            axis.set_ylim(0, 1)
            axis.grid(axis="y", alpha=0.2)
            axis.set_axisbelow(True)
        axes[i, 0].set_ylabel(title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside upper center", frameon=False, ncols=3)
    figure.supxlabel("Visible component area (px²)")
    figure.savefig(out / "three_model_defect_sizes.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    out = parser.parse_args().run_dir.resolve()
    run.require(out.parent == run.HERE / "runs", "Expected a local U-Net run")
    result = json.loads((out / "report.json").read_text(encoding="utf-8"))
    run.require(result["baseline_report_hashes"] == {"patchcore": run.sha(run.PC_REPORT), "anomalydino": run.sha(run.DINO_REPORT)},
                "Saved baseline differs from the evaluation reference")
    pc, dino = [json.loads(path.read_text(encoding="utf-8"))["lighting"] for path in (run.PC_REPORT, run.DINO_REPORT)]
    unet = result["lighting"]
    models = (pc, dino, unet)
    write_table(out, models)
    loss_curves(out, unet)
    metric_charts(out, models)
    annotated_figures(out, unet, run.load_audit()[0])
    print(f"Wrote U-Net visual results and three-model comparisons to {out}", flush=True)


if __name__ == "__main__":
    main()
