from __future__ import annotations

import hashlib
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


REQUIRED_CHECKPOINT_KEYS = {
    "model",
    "optimizer",
    "scheduler",
    "amp_grad_scaler",
    "completed_epoch",
    "best_metric",
    "best_model_state",
    "rng_state",
    "dataloader_state",
    "config",
    "git",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(repo_root: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
        ).strip()

    source_digest = hashlib.sha256()
    source_paths = []
    for pattern in ("src/**/*.py", "scripts/*.py", "configs/*.json", "environment/*.json"):
        source_paths.extend(repo_root.glob(pattern))
    for path in sorted(set(source_paths)):
        relative = path.relative_to(repo_root).as_posix().encode("utf-8")
        source_digest.update(len(relative).to_bytes(4, "big"))
        source_digest.update(relative)
        source_digest.update(path.read_bytes())

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "worktree_status": run("status", "--short"),
        "diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"], cwd=repo_root)
        ).hexdigest(),
        "source_tree_sha256": source_digest.hexdigest(),
    }


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def validate_checkpoint_payload(payload: dict[str, Any]) -> None:
    missing = REQUIRED_CHECKPOINT_KEYS - payload.keys()
    if missing:
        raise ValueError(f"Checkpoint missing required fields: {sorted(missing)}")
    if not isinstance(payload["completed_epoch"], int) or payload["completed_epoch"] < 1:
        raise ValueError("completed_epoch must be a positive integer")
    dictionary_fields = (
        "model", "optimizer", "scheduler", "best_model_state", "rng_state",
        "dataloader_state", "config", "git",
    )
    for key in dictionary_fields:
        if not isinstance(payload[key], dict):
            raise ValueError(f"Checkpoint field {key!r} must be a dictionary")


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location)
    validate_checkpoint_payload(payload)
    return payload


def atomic_save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validate_checkpoint_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_hash = sha256_file(temporary)
        verified = load_checkpoint(temporary, map_location="cpu")
        if verified["completed_epoch"] != payload["completed_epoch"]:
            raise ValueError("Checkpoint epoch changed during verification")
        os.replace(temporary, path)
        final_hash = sha256_file(path)
        if final_hash != temporary_hash:
            raise IOError("Atomic checkpoint replacement failed integrity verification")
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": final_hash}
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
