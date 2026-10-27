from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_anomaly.baseline import load_upstream, seed_all
from medical_anomaly.checkpointing import sha256_file
from medical_anomaly.diagnostics import sample_id, validate_alignment
from medical_anomaly.explicit_splits import manifest_argument


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


@torch.inference_mode()
def collect(model, loader, device: str, data_root: Path) -> dict[str, np.ndarray]:
    map_logits, score_logits, labels = [], [], []
    identifiers, image_paths, mask_paths, source_sizes = [], [], [], []
    model.eval()
    for batch in loader:
        dense, head = model(batch["image"].to(device))
        dense_array = dense.detach().cpu().numpy()
        if dense_array.ndim != 4 or dense_array.shape[1] != 1:
            raise ValueError(f"unexpected dense output shape: {dense_array.shape}")
        map_logits.append(dense_array[:, 0].astype(np.float32, copy=False))
        score_logits.append(head.detach().cpu().numpy().astype(np.float32, copy=False))
        labels.append(batch["label"].numpy().astype(np.int8, copy=False))
        batch_paths = list(batch["image_path"])
        batch_masks = list(batch.get("mask_path", [""] * len(batch_paths)))
        for image_path, mask_path in zip(batch_paths, batch_masks):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise OSError(f"cannot decode image metadata: {image_path}")
            identifiers.append(sample_id(image_path, data_root))
            image_paths.append(str(Path(image_path).resolve()))
            mask_paths.append(str(Path(mask_path).resolve()) if mask_path else "")
            source_sizes.append(image.shape)
    result = {
        "sample_ids": np.asarray(identifiers),
        "labels": np.concatenate(labels),
        "score_logits": np.concatenate(score_logits),
        "map_logits": np.concatenate(map_logits),
        "image_paths": np.asarray(image_paths),
        "mask_paths": np.asarray(mask_paths),
        "source_sizes": np.asarray(source_sizes, dtype=np.int32),
    }
    validate_alignment(result["sample_ids"], result["labels"], result["score_logits"], result["map_logits"])
    return result


def write_bundle(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as loaded:
            for split in ("validation", "test"):
                validate_alignment(
                    loaded[f"{split}_sample_ids"], loaded[f"{split}_labels"],
                    loaded[f"{split}_score_logits"], loaded[f"{split}_map_logits"],
                )
        temporary_hash = sha256_file(temporary)
        os.replace(temporary, path)
        final_hash = sha256_file(path)
        if final_hash != temporary_hash:
            raise IOError("prediction bundle changed during atomic replacement")
        return {"path": str(path), "bytes": path.stat().st_size, "sha256": final_hash}
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-archive", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    seed_all(config["seed"])
    output = args.output or args.run_directory / "predictions.npz"
    manifest_path = args.manifest or args.run_directory / "prediction_manifest.json"
    weights = args.run_directory / "checkpoints" / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(weights)

    SuperSimpleNet, _, _ = load_upstream(args.upstream_root)
    from medical_anomaly.data.ds1_upstream import DS1UpstreamDataModule

    dm = DS1UpstreamDataModule(
        args.data_root, args.split_archive, tuple(config["image_size"]), config["batch_size"],
        config["num_workers"], config["val_fraction"], config["seed"],
        config.get("supervision_mode", "mixed"),
        split_manifest=manifest_argument(args, config), manifest_sha256=config.get("split_manifest_sha256"),
    )
    dm.setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperSimpleNet(tuple(config["image_size"]), config).to(device)
    model.load_model(weights)
    splits = {
        "validation": collect(model, dm.val_dataloader(), device, args.data_root),
        "test": collect(model, dm.test_dataloader(), device, args.data_root),
    }
    arrays = {
        f"{split}_{key}": value
        for split, values in splits.items()
        for key, value in values.items()
    }
    bundle = write_bundle(output, arrays)
    manifest = {
        "schema_version": 1,
        "experiment": config["name"],
        "seed": config["seed"],
        "config_path": str(args.config),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "source_commit": git_commit(),
        "device": device,
        "weights": {"path": str(weights), "bytes": weights.stat().st_size, "sha256": sha256_file(weights)},
        "bundle": bundle,
        "splits": {
            split: {
                "samples": len(values["labels"]),
                "normal": int((values["labels"] == 0).sum()),
                "defect": int((values["labels"] == 1).sum()),
                "map_shape": list(values["map_logits"].shape),
                "head_shape": list(values["score_logits"].shape),
                "source_size_values": np.unique(values["source_sizes"], axis=0).tolist(),
            }
            for split, values in splits.items()
        },
        "tensor_semantics": {
            "score_logits": "raw SuperSimpleNet classification-head anomaly logit; higher is more anomalous",
            "map_logits": "raw post-bilinear-resize and post-Gaussian-blur dense anomaly logit at model input size",
            "activation": "sigmoid is applied by analysis, not stored redundantly",
        },
        "command": [sys.executable, *sys.argv],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
