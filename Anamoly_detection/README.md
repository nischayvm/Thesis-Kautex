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

Double-click [Render_ANNAlyzer.bat](Render_ANNAlyzer.bat) from the project root. It starts or resumes the dataset at `data/annalyzer_batch_2048`, shows progress and ETA in the terminal, and writes a timestamped log in `data`. Resume requires the same scene, scripts, and seed. Use the completed batch for training. The earlier prototype and timing pilot folders under `data` are separate.

The batch used Cycles with **100 maximum samples, 0.015 adaptive noise threshold, denoising, and RTX 4060 OptiX**. The completed run recorded **about 4 h 30 min** for 501 renders. The regular clean reference was rendered once, and the masks were exported with the defective images.
