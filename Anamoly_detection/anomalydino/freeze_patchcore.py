"""Fingerprint the completed PatchCore implementation, environment and results."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCHCORE = HERE.parent / "patchcore"
MANIFEST = HERE / "patchcore_freeze_v1.json"
RUNS = (
    "regular_smoke_20261124T072401_611224Z",
    "red_phase3_20261124T075750_651308Z",
    "lighting_benchmark_20261124T084630_908213Z",
)


def fingerprint(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def capture():
    project = [p for p in PATCHCORE.iterdir() if p.suffix in {".py", ".json", ".toml", ".lock", ".pth"}]
    environment = PATCHCORE / ".venv"
    metadata = list((environment / "Lib" / "site-packages").glob("*.dist-info/METADATA"))
    if not metadata:
        raise RuntimeError("PatchCore environment metadata unavailable")
    results = []
    for run in RUNS:
        folder = PATCHCORE / "runs" / run
        if not (folder / "report.json").is_file():
            raise RuntimeError(f"PatchCore run {run} is missing")
        results.extend(path for path in folder.rglob("*") if path.is_file())
    paths = sorted(project + [environment / "pyvenv.cfg", *metadata] + results)
    return {
        "algorithm": "sha256",
        "scope": "PatchCore project, package metadata, weights and all files in the three completed runs",
        "files": {path.relative_to(PATCHCORE).as_posix(): fingerprint(path) for path in paths},
    }


if __name__ == "__main__":
    current = capture()
    if MANIFEST.exists():
        frozen = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if current != frozen:
            raise RuntimeError("PatchCore differs from the frozen manifest")
        print(f"Verified {len(current['files'])} frozen PatchCore files")
    else:
        with MANIFEST.open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(current, indent=2) + "\n")
        print(f"Froze {len(current['files'])} PatchCore files in {MANIFEST}")
