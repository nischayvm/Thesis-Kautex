"""Snapshot AnomalyDINO before running the supervised benchmark."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DINO = HERE.parent / "anomalydino"
RUN = DINO / "runs" / "anomalydino_20261124T091730_300718Z"
MANIFEST = HERE / "anomalydino_freeze_v1.json"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def capture():
    environment = DINO / ".venv"
    files = [p for p in DINO.iterdir() if p.is_file()]
    files.extend((environment / "Lib" / "site-packages").glob("*.dist-info/METADATA"))
    files.append(environment / "pyvenv.cfg")
    files.extend((DINO / ".hf-cache").rglob("model.safetensors"))
    files.extend(p for p in RUN.rglob("*") if p.is_file())
    if not (RUN / "report.json").is_file() or len(files) < 100:
        raise RuntimeError("Completed AnomalyDINO run or dependencies are unavailable")
    return {"algorithm": "sha256", "files": {p.relative_to(DINO).as_posix(): digest(p) for p in sorted(files)}}


if __name__ == "__main__":
    snapshot = capture()
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text(encoding="utf-8")) != snapshot:
            raise RuntimeError("AnomalyDINO differs from the frozen manifest")
        print(f"Verified {len(snapshot['files'])} frozen AnomalyDINO files")
    else:
        with MANIFEST.open("x", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(snapshot, indent=2) + "\n")
        print(f"Froze {len(snapshot['files'])} AnomalyDINO files")
