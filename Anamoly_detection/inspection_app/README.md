# ANNAlyzer tray inspection

Double-click [Run_Tray_Inspection.bat](../../bat%20files/Run_Tray_Inspection.bat). Click **Open image…** to choose an opaque RGB Blender PNG. You can also drop the file into the image field or paste its path and press Enter. Select a model and click **Inspect**. The status line shows checkpoint verification, model loading, and inference as they happen. The result box shows **PASS**, **FAIL**, or **SCORE ONLY** with the raw image score, saved validation threshold, and inference time.

After a prediction, click **Open mask…** to choose the matching binary PNG. Drag-and-drop and path entry work here too. Use an opaque, black-and-white mask at the input image's original dimensions. White marks defect pixels. An all-black mask is valid for a clean image. Check the tray, prediction overlay, and ground-truth overlay in the same window.

Use the **Prediction** dropdown to switch between the overlay on the tray and the standalone heatmap. Both use the same color scale and continuous scores. Scroll over any displayed image to zoom, then drag to pan. All three panels show the same region. Use **Fit** to reset the view.

The comparison shows whether the image decision agrees with the mask and computes pixel AP and pixel AUROC on the full frame using continuous scores before heatmap coloring. Pixel ranking metrics require both defect and background pixels. The V3 reports supply image-level thresholds. Dice and IoU require a separate calibrated pixel cutoff. A mask leaves the score and decision unchanged, and a new image clears the previous comparison.

Click **Show pipeline** after loading an image to see all 25 overlapping tiles. Select a tile to view the model's spatial grid or magnify a sample of individual input pixels. The panel shows PatchCore's feature grids, AnomalyDINO's padded transformer patches, or the U-Net encoder bottleneck. It uses the same window as the prediction overlays. Gray bands illustrate padding, and grid lines mark spatial layout. The model's scored output appears in the prediction overlay.

For a benchmark-matched demo, render at 2048 × 2048 with the same Front Left camera, framing, and lighting. Other sizes are accepted. The app fits them into the model's 2048 × 2048 input using aspect-preserving resizing and edge padding as needed. It crops the padding from the score map and aligns the continuous scores back to the original frame for display and mask comparison. Decisions on resized inputs are experimental because the saved thresholds come from native 2048 × 2048 renders. Save fresh renders outside the benchmark dataset.

**Automatic** estimates lighting from the strength and prevalence of red and green casts in sampled RGB pixels. It works from the image itself, independently of the defect model. For mixed, dark, or ambiguous frames, choose the known render lighting manually. Red and green select the corresponding same-light model. Regular selects the regular PatchCore or AnomalyDINO state, or the red-to-regular U-Net transfer protocol.

The app loads the checkpoint identified in the latest completed local V3 validation report for the selected model and lighting. It verifies the checkpoint hash and keeps one worker loaded for repeated images. PatchCore, AnomalyDINO, and U-Net run in their existing CUDA model environments. Each worker reuses the benchmark's inference and stitching. The display uses that run's fixed validation heatmap scale. **FAIL** means the image score is strictly above the validation threshold. The input copy and overlays live in a local temporary folder for the app session.

## Setup on another Windows machine

Ensure the three model environments, local pretrained weights where required, and completed V3 run folders are present. For the UI, from the repository root with Python 3.10 available, run:

```powershell
python -m venv Anamoly_detection/inspection_app/.venv
Anamoly_detection/inspection_app/.venv/Scripts/python.exe -m pip install -r Anamoly_detection/inspection_app/requirements.txt
```
