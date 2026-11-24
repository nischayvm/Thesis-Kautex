"""Checkpoint helper copied from the frozen PatchCore benchmark."""

import hashlib

import torch
from smoke import MIB, require

def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_model(model, path):
    torch.save(model.state_dict(), path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    require(torch.equal(saved["memory_bank"], model.memory_bank.cpu()), f"Checkpoint bank mismatch for {path.name}")
    return {"path": path.relative_to(path.parent.parent).as_posix(), "sha256": sha256(path), "dimensions": list(model.memory_bank.shape),
            "bank_mib": model.memory_bank.nbytes / MIB}
