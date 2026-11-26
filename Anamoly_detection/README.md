# ANNAlyzer battery tray inspection dataset

This thesis project uses Blender to simulate a factory inspection cell and generate labeled images for blister detection. The model is a generic synthetic battery tray based on visual references. The completed batch is in [data/annalyzer_batch_2048](data/annalyzer_batch_2048).

## Blender scene

- **Tray:** A 1500 × 700 mm injection-moulded automotive battery tray with drafted walls, reinforcing ribs, mounting bosses, and a wide flange with fastener holes. The shell uses a dark, matte PA6-GF30 material. The floor simulates a textured cross-ply laminate overmoulded by the ribs and walls.
- **Defects:** Small irregular flange blisters, including near fastener holes, and broader lifts in the cross-ply laminate. Some laminate lifts span thin ribs. Both use raised geometry. The scene can also generate curved-edge burns and sink marks. The released dataset contains blisters only.
- **Inspection cell:** A factory bay with diffused ceiling LEDs, a white worktop with direction labels, an adjustable T-slot rig, four inspection cameras, four LED bars, and an ANNAlyzer workstation. The Blender viewport sidebar has light color, Kelvin, intensity, dark-field, camera, and four-camera capture controls. The dataset batch uses only the Front Left inspection camera.

| File | Purpose |
|---|---|
| [battery_tray_base.blend](synthetic_dataset/battery_tray_base.blend) | Standalone base tray |
| [battery_tray_inspection_cell.blend](synthetic_dataset/battery_tray_inspection_cell.blend) | Clean inspection-cell scene |
| [battery_tray_inspection_anomalies.blend](synthetic_dataset/battery_tray_inspection_anomalies.blend) | Working scene used for the batch |
| [generate_tray_anomalies.py](scripts/generate_tray_anomalies.py) | Seeded defect generation. The batch uses the copy embedded in the working scene. |
| [build_crossply_material.py](scripts/build_crossply_material.py) | Procedural cross-ply material |
| [annalyzer_controls.py](scripts/annalyzer_controls.py) | Blender sidebar and interactive four-camera capture |
| [render_annalyzer_batch.py](scripts/render_annalyzer_batch.py) | Dataset generation, checkpoints, and progress logging |
| [render_prototype_dataset.py](scripts/render_prototype_dataset.py) | Earlier prototype runner and the PNG and mask helpers reused by the batch |

## Completed batch

There are **300 clean images, 300 blistered images, and 300 binary masks**, all **2048 × 2048, 8-bit PNG**. Each lighting set has 100 matched examples. Within a pair, the scene, camera, lighting, and materials are identical. A new seed varies blister position, size, shape, and count. Each defective tray contains **7–10 flange blisters** and **2–8 cross-ply blisters**. Some blisters may be outside the image or occluded.

| Set | Lighting and placement |
|---|---|
| `regular` | Default rig and ceiling lighting. Camera at 1.61 m and LEDs at 1.102 m. The 100 normal files are copies of one clean reference. |
| `red` | Red rig LEDs. 34 low, 33 mid, and 33 high setups. Both camera and LED heights vary within the bands. |
| `green` | Green rig LEDs. The same height and intensity schedule as red, with independent blister seeds. |

In the red and green sets, nominal camera heights range from **1.28 to 1.61 m** and nominal LED heights from **1.102 to 1.60 m**. Rig intensity varies from **1.50 to 2.00×** and ceiling intensity from **0.35 to 1.10×** before dark-field dimming. Each color has **25 partial dark-field setups** with a mix of 0.65–0.90. This lowers the LED bars, narrows and tilts the beams, and dims the ceiling. Use `effective_led_height_m` for the evaluated LED height.

```text
data/annalyzer_batch_2048/
  dataset.json
  samples.json
  regular/{normal,bubbles,masks}/0001.png ... 0100.png
  red/{normal,bubbles,masks}/0001.png ... 0100.png
  green/{normal,bubbles,masks}/0001.png ... 0100.png
```

- **`dataset.json`** records completion, the render settings, and hashes of the scene and scripts.
- **`samples.json`** is keyed by `color/sample_id`, for example `red/0001`. Each record has a `files` object with dataset-relative paths for `normal`, `bubbles`, and `masks`. It also records the seed, height band, heights, intensities, dark-field mix, counts, and geometry metadata in `instances`. Instance centers and radii use tray-local metres. They are 3D metadata.
- **Masks** are pixel-aligned with the matching `bubbles` image. They are single-channel PNGs with **0 for background and 255 for visible blister pixels**. Flange and cross-ply blisters share the same foreground class. The compositor creates the mask from Blender's `Defect_Blister` shader pass in the same render. Each mask was checked for binary values and nonzero foreground.

## Loading data for ML

1. Read `samples.json` as UTF-8 and resolve `files` relative to the dataset root. Each key provides one normal RGB image and one anomalous RGB image. Read the anomalous image's mask as grayscale and use `mask > 0` for a binary target. For supervised training with normal images, create an all-zero target at load time.
2. Use the regular set as the controlled baseline and report results on the red and green sets separately. The regular set has one unique normal view. Deduplicate those normals and avoid counting copies as independent test images.
3. Build the split before creating crops or augmentations. Keep each clean image, blistered image, and mask in the same split. Keep the red and green records with the same numeric ID together when splitting the colored data, since those records share the camera and lighting setup apart from color. To test generalization across colors, hold out an entire color intentionally. Apply the same spatial transforms to image and mask, with nearest-neighbor resizing for masks.
4. Initial baselines are **PatchCore** and **AnomalyDINO** using normal training images, and **U-Net** using the binary segmentation masks. The dataset contains synthetic renders from one tray design. Results measure performance on this synthetic inspection setup.

## Running the batch

The render launcher opens a job menu for the original V1 three-light batch, the V2 regular-light normal supplement, and saved status. V1 writes to `data/annalyzer_batch_2048` and V2 to `data/annalyzer_regular_normals_v2`. Completed datasets are reported as complete before Blender launches. An incomplete run can resume with the same scene, scripts and seed. [Render launcher](../bat%20files/Render_ANNAlyzer.bat)

The batch used Cycles with **100 maximum samples, 0.015 adaptive noise threshold, denoising, and RTX 4060 OptiX**. The completed run recorded **about 4 h 30 min** for 501 renders. The regular clean reference was rendered once, and the masks were exported with the defective images.

## Regular-light V2 models

Each model folder contains frozen source and configuration under `V1/`, the adapted workflow under `V2/`, and outputs under `results/V1/` and `results/V2/`. The V1 scripts preserve the original layout assumptions for historical reproducibility. The batch launchers use the existing per-model `.venv` at the model root.

Open a launcher in the `bat files` folder. [PatchCore](../bat%20files/Run_PatchCore_V2.bat) · [AnomalyDINO](../bat%20files/Run_AnomalyDINO_V2.bat) · [U-Net](../bat%20files/Run_UNet_V2.bat)

| Command argument | Action |
| --- | --- |
| `check-env` | Verify the installed CUDA environment and required weights. |
| `check` | Verify the completed V2 normal dataset, frozen split and fixed-pose validation pairs. |
| Double-click or no argument | Run the smoke check and validation using the reviewed V2 split. Fit PatchCore or AnomalyDINO from the 70 training normals, or load the frozen red U-Net. |
| `validate` | Run the same validation workflow with the review decision supplied on the command line. |
| `test "<validation-run-folder>"` | Reload a completed run's checkpoint and apply the saved validation threshold to its reserved test cohorts. |
| `all` | Explicitly request validation followed by reserved test evaluation. |

Use `--help` for launcher options. The double-click default uses the decision recorded for the fingerprinted V2 manifest and duplicate audit. To reuse that decision with an explicit command, add `--use-reviewed-split`. For a changed dataset, review the newly flagged pairs before using `--accept-near-duplicates`.

V2 uses the 15 varied-pose validation normals to set a threshold at their maximum score. A score above that threshold counts as an alarm. The 15 varied-pose test normals measure false alarms. Fixed-pose defects use the V1 assignments of three validation and 97 reserved test views and provide separate sensitivity and localization estimates. Varied-pose image AUROC and AP are unavailable because those captures have no matched defects. The red and green results come from the V1 benchmark. [Research journal](JOURNAL.md)

After validation, double-click the publishing batch to choose an available version and toggle its completed lighting conditions. It selects the latest completed validation per model and light, appends new selected results to [JOURNAL.md](JOURNAL.md), and pushes the journal and results to GitHub `main`. Git tracks reports, score CSVs, and score plots. Git LFS stores fitted states, available diagnostics, and heatmaps. Source images, masks, raw score arrays, and test artifacts stay local. From the project root in PowerShell, run `& ".\bat files\Push_Model_Results.bat" --dry-run --version V3 --lighting red green` to preview an explicit selection. [Publishing launcher](../bat%20files/Push_Model_Results.bat)

Run `python scripts/update_journal_from_results.py` independently to append summaries from completed local V2–V4 reports. Add `--from-git` to read reports published on GitHub `main` or `--dry-run` to preview the journal entry. The importer skips previously recorded runs and keeps held-out test results separate from validation. [Importer](../scripts/update_journal_from_results.py)

## Local benchmark report

From the project root, use the PatchCore environment to generate the [interactive benchmark report](reports/index.html).

```powershell
.\Anamoly_detection\patchcore\.venv\Scripts\python.exe -B .\scripts\generate_benchmark_html_report.py
```

Open `Anamoly_detection/reports/index.html` in a browser. The report discovers versioned result folders, selects completed runs and compares exact source-image and mask pairs across models. The **Compare** tab reports metric changes for verified matching cohorts and shows the original, ground truth and model outputs for the same frame. Use the run selectors for earlier runs and the gallery for all saved images. If needed, the generator creates display-only previews from saved score maps and verified original data, using the recorded run scale. It writes the page and its media cache in `Anamoly_detection/reports/` and links to the existing full-size artifacts. [Generator](../scripts/generate_benchmark_html_report.py)

### Excel workbooks

Run `python scripts/generate_benchmark_excel.py` from the project root with a Python environment that has `openpyxl`. It writes one `Vn_results.xlsx` per discovered version to `Anamoly_detection/reports/excel/`. Add `--versions V3 V4` to select specific versions. The workbooks contain editable charts, filterable results, per-image scores, resource and training data, numeric JSON fields, and links to saved artifacts. [Excel exporter](../scripts/generate_benchmark_excel.py)

## V3 three-lighting runs

The existing batch filenames launch V3. Double-click PatchCore or AnomalyDINO in `bat files/` to select lighting. The checklist starts with all three selected. Type `1`, `2`, or `3` and press Enter to toggle red, green, or regular. `A` toggles all, an empty Enter starts the selected run, and `Q` cancels. To skip the menu, pass `validate` for all lights or `validate --lighting red green` for a subset. U-Net still runs all three directly. Pass `check` to verify the frozen datasets and `check-env` to verify the model environment. The explicit `v2` selector invokes the previous regular-only workflow. For example, `Run_PatchCore_V2.bat v2 validate --use-reviewed-split` runs V2. [PatchCore](../bat%20files/Run_PatchCore_V2.bat) · [AnomalyDINO](../bat%20files/Run_AnomalyDINO_V2.bat) · [U-Net](../bat%20files/Run_UNet_V2.bat)

V3 refits separate PatchCore and AnomalyDINO normal banks for each light using the existing V1 red/green splits and V2 regular normal split. U-Net fits separate supervised red and green checkpoints and transfers the selected V3 red checkpoint to regular. Red and green each validate on 15 clean and 15 defective frames. Regular uses 15 varied clean views for calibration and the same three fixed-pose defects for localization. V3 run folders are local under each model’s `results/V3/`. Regenerate the HTML report after a run finishes to see its V3 tab. [V3 entry point](v3_workflow.py)
