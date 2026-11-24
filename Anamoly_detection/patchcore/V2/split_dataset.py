"""Freeze the ANNAlyzer Phase 3 setup groups before reading image pixels."""

import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1] / "dataset_audit_manifest_v1.jsonl"
SPLIT = HERE.parent / "V1" / "split_v1.json"
SEED = 24117
FOLDS = ("train", "validation", "test")
COLOURS = ("red", "green")
REGULAR_VALIDATION = ("0004", "0074", "0097")
NEAR_PAIR = ("0091", "0094")


def load_audit():
    content = AUDIT.read_bytes()
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines()]
    records = {row["sample_id"]: row for row in rows}
    if len(rows) != 300 or len(records) != 300:
        raise ValueError("Expected 300 distinct audited triples")
    return records, hashlib.sha256(content).hexdigest()


def build_split(records, audit_digest):
    ids = [f"{number:04d}" for number in range(1, 101)]
    strata = {(band, dark): [] for band in ("low", "mid", "high") for dark in (False, True)}
    for sample in ids:
        red, green = (records[f"{colour}/{sample}"] for colour in COLOURS)
        if red["capture_setup_id"] != green["capture_setup_id"] or red["render_setup"] != green["render_setup"]:
            raise ValueError(f"Mismatched red/green setup {sample}")
        setup = red["render_setup"]
        strata[setup["height_band"], bool(setup["dark_field_mix"])].append(sample)

    rng = random.Random(SEED)
    folds = {fold: [] for fold in FOLDS}
    stratum_counts = {}
    for (band, dark), original in strata.items():
        size = 1 if dark else 4
        if any(sample in original for sample in NEAR_PAIR) and not all(sample in original for sample in NEAR_PAIR):
            raise ValueError("The audited near-duplicate pair has different strata")
        pool = original.copy()
        for _ in range(1000):
            rng.shuffle(pool)
            local = {"train": pool[2 * size:], "validation": pool[:size], "test": pool[size:2 * size]}
            if not all(sample in original for sample in NEAR_PAIR) or any(set(NEAR_PAIR) <= set(group) for group in local.values()):
                break
        else:
            raise RuntimeError("Unable to keep the audited near-duplicate pair grouped")
        for fold, group in local.items():
            folds[fold].extend(group)
        stratum_counts[f"{band}_{'dark' if dark else 'standard'}"] = {fold: len(group) for fold, group in local.items()}

    if [len(folds[fold]) for fold in FOLDS] != [70, 15, 15] or sorted(sum(folds.values(), [])) != ids:
        raise ValueError("Unexpected coloured split sizes or overlapping IDs")
    digest_folds = {}
    for fold, group in folds.items():
        for sample in group:
            for colour in COLOURS:
                for asset in records[f"{colour}/{sample}"]["files"].values():
                    digest = asset["decoded_pixel_sha256"]
                    if digest_folds.setdefault(digest, fold) != fold:
                        raise ValueError(f"Exact pixel duplicate across folds for {colour}/{sample}")

    normals = [records[f"regular/{sample}"]["files"]["normal"] for sample in ids]
    if len({asset["decoded_pixel_sha256"] for asset in normals}) != 1:
        raise ValueError("Expected one unique regular normal reference")
    return {
        "schema": 1,
        "seed": SEED,
        "audit_manifest_sha256": audit_digest,
        "method": "setup-ID grouping and fixed quotas by height band and dark-field status",
        "colours_sharing_subset": list(COLOURS),
        "coloured_setup_ids": {fold: sorted(group) for fold, group in folds.items()},
        "stratum_counts": stratum_counts,
        "preserved_near_duplicate_group": list(NEAR_PAIR),
        "regular": {
            "reference_sample_id": "regular/0001",
            "reference_normal": normals[0]["path"],
            "normal_aliases_excluded": 99,
            "normal_validation": [],
            "normal_test": [],
            "defective_validation_ids": [f"regular/{sample}" for sample in REGULAR_VALIDATION],
            "defective_test_ids": [f"regular/{sample}" for sample in ids if sample not in REGULAR_VALIDATION],
        },
    }


def split_text(records, digest):
    return json.dumps(build_split(records, digest), indent=2) + "\n"
