import random

import numpy as np
import torch

import medical_anomaly.checkpointing as checkpointing
from medical_anomaly.checkpointing import atomic_save_checkpoint, load_checkpoint


def test_atomic_checkpoint_round_trip(tmp_path):
    path = tmp_path / "latest.ckpt"
    payload = {
        "model": {"weight": torch.tensor([1.0])},
        "optimizer": {},
        "scheduler": {},
        "amp_grad_scaler": None,
        "completed_epoch": 3,
        "best_metric": {"value": 1.0},
        "best_model_state": {"weight": torch.tensor([1.0])},
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
        },
        "dataloader_state": {"generator": torch.Generator().get_state()},
        "config": {"epochs": 30},
        "git": {"commit": "test"},
    }
    info = atomic_save_checkpoint(path, payload)
    restored = load_checkpoint(path)
    assert restored["completed_epoch"] == 3
    assert info["bytes"] == path.stat().st_size
    assert not path.with_name("latest.ckpt.tmp").exists()


def test_failed_verification_preserves_previous_latest(tmp_path, monkeypatch):
    path = tmp_path / "latest.ckpt"
    payload = {
        "model": {"weight": torch.tensor([1.0])}, "optimizer": {}, "scheduler": {},
        "amp_grad_scaler": None, "completed_epoch": 1,
        "best_metric": {"value": 1.0}, "best_model_state": {"weight": torch.tensor([1.0])},
        "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch_cpu": torch.get_rng_state(), "torch_cuda": []},
        "dataloader_state": {"generator": torch.Generator().get_state()},
        "config": {"epochs": 30}, "git": {"commit": "test"},
    }
    original = atomic_save_checkpoint(path, payload)
    payload["completed_epoch"] = 2

    def fail_verification(*_args, **_kwargs):
        raise ValueError("simulated verification failure")

    monkeypatch.setattr(checkpointing, "load_checkpoint", fail_verification)
    try:
        atomic_save_checkpoint(path, payload)
    except ValueError:
        pass
    else:
        raise AssertionError("verification failure must propagate")
    assert checkpointing.sha256_file(path) == original["sha256"]
    assert not path.with_name("latest.ckpt.tmp").exists()
