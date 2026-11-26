"""Audit custom paired renders and split capture groups before tiling."""

import hashlib
import random
from pathlib import Path

from workstation_config import custom_dataset, read_json, write_json


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(config, destination):
    from PIL import Image

    dataset = custom_dataset(config["dataset"])
    status = read_json(dataset / "dataset.json")
    if not status.get("complete") or status["configuration"].get("workstation_render") != config["render"]:
        raise ValueError("The custom render is incomplete or its settings differ from this job.")
    settings = config["render"]
    catalog = read_json(dataset / "samples.json")
    count = settings["pairs"]
    ids = [f"{i:04}" for i in range(1, count + 1)]
    expected = {f"{light}/{sample}" for light in settings["lights"] for sample in ids}
    if set(catalog) != expected:
        raise ValueError("The render catalog has missing or unexpected pairs.")

    split = config["split"]
    shuffled = ids.copy()
    random.Random(split["seed"]).shuffle(shuffled)
    train = min(count - 2, max(1, round(count * split["train"] / 100)))
    validation = min(count - train - 1, max(1, round(count * split["validation"] / 100)))
    folds = {label: sorted(values) for label, values in
             (("train", shuffled[:train]), ("validation", shuffled[train:train + validation]),
              ("test", shuffled[train + validation:]))}
    assignment = {sample: fold for fold, values in folds.items() for sample in values}
    hashes, previews, records, candidates = {}, {}, {}, []
    for index, (key, entry) in enumerate(sorted(catalog.items()), 1):
        light, sample = key.split("/")
        fold = assignment[sample]
        files = {}
        for kind, mode in (("normal", "RGB"), ("bubbles", "RGB"), ("masks", "L")):
            path = (dataset / entry["files"][kind]).resolve()
            if not path.is_relative_to(dataset) or path != dataset / light / kind / f"{sample}.png":
                raise ValueError(f"Unexpected asset path for {key}: {path}")
            with Image.open(path) as image:
                if image.size != (settings["resolution"],) * 2 or image.format != "PNG" or image.mode != mode:
                    raise ValueError(f"Invalid PNG dimensions or encoding: {path}")
                if kind == "masks":
                    counts = image.histogram()
                    if (sum(counts[1:255]) or not counts[0] or not counts[255]
                            or counts[255] != entry["mask_positive_pixels"]):
                        raise ValueError(f"Invalid or misreported binary mask: {path}")
                else:
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    if digest in hashes:
                        raise ValueError(f"Identical RGB renders: {hashes[digest]} and {path}")
                    hashes[digest] = str(path)
                    pixels = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
                    fingerprint = sum((pixels[9*y+x] > pixels[9*y+x+1]) << (8*y+x)
                                      for y in range(8) for x in range(8))
                    for other_fold, other_key, other_hash in previews.get((light, kind), []):
                        if other_fold != fold and (other_hash ^ fingerprint).bit_count() <= 1:
                            candidates.append([other_key, key, kind])
                    previews.setdefault((light, kind), []).append((fold, key, fingerprint))
            files[kind] = {"path": str(path), "sha256": sha(path),
                           "label": "normal" if kind == "normal" else "anomalous"}
        records[key] = {"sample_id": key, "lighting": light, "files": files,
                        "mask_positive_pixels": entry["mask_positive_pixels"]}
        if index % 25 == 0 or index == len(catalog):
            print(f"Audited {index}/{len(catalog)} image and mask pairs", flush=True)
    manifest = {"schema": 1, "dataset": str(dataset), "catalog_sha256": sha(dataset / "samples.json"),
                "configuration_sha256": sha(dataset / "dataset.json"),
                "split_settings": {k: v for k, v in split.items() if k != "accept_near_duplicates"},
                "folds": folds, "records": records, "near_duplicate_candidates": candidates}
    if Path(destination).exists():
        if read_json(destination) != manifest:
            raise ValueError("An audit already exists with different settings or data. Save a new job config.")
    else:
        write_json(destination, manifest)
    if candidates and not split["accept_near_duplicates"]:
        print(f"Review {len(candidates)} cross-fold image similarity candidates in {destination}.", flush=True)
    print(f"Grouped split: { {k: len(v) for k, v in folds.items()} }", flush=True)
    return manifest


def get_records(manifest, light, fold):
    return [manifest["records"][f"{light}/{sample}"] for sample in manifest["folds"][fold]]
