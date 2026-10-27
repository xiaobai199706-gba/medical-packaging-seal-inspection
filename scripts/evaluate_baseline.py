from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_anomaly.baseline import best_threshold, evaluate, load_upstream, predict, save_error_panels, save_json, seed_all
from medical_anomaly.data import MVTecPillDataModule


def progress_writer(path: Path | None, stage: str):
    def write(completed: int, total: int) -> None:
        if path is None:
            return
        payload = {"stage": stage, "completed": completed, "total": total, "updated_unix": time.time()}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    return write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-archive", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--skip-panels", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); seed_all(config["seed"])
    SuperSimpleNet, _, _ = load_upstream(args.upstream_root)
    if config["name"].startswith("mvtec"):
        dm = MVTecPillDataModule(args.data_root, config["batch_size"], config["seed"], config["train_limit"], config["eval_limit"])
    else:
        from medical_anomaly.data.ds1_upstream import DS1UpstreamDataModule
        dm = DS1UpstreamDataModule(args.data_root, args.split_archive, tuple(config["image_size"]),
                                   config["batch_size"], config["num_workers"], config["val_fraction"], config["seed"],
                                   config.get("supervision_mode", "mixed"),
                                   split_manifest=__import__('medical_anomaly.explicit_splits', fromlist=['manifest_argument']).manifest_argument(args, config),
                                   manifest_sha256=config.get("split_manifest_sha256"))
        if config.get("protocol") == "SSI-ContentGroup-Sparse3-v1":
            raise ValueError("Use run_content_group_experiment.py evaluate for the original-scale group protocol")
    dm.setup(); device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperSimpleNet(tuple(config["image_size"]), config).to(device)
    model.load_model(args.run_directory / "checkpoints" / "weights.pt")
    validation = predict(model, dm.val_dataloader(), device, progress_writer(args.progress_file, "validation_inference"))
    test = predict(model, dm.test_dataloader(), device, progress_writer(args.progress_file, "test_inference"))
    image_threshold = best_threshold(validation["labels"], validation["scores"], "youden")
    pixel_threshold = best_threshold(validation["masks"].ravel(), validation["maps"].ravel(), "dice")
    metrics = evaluate(test, image_threshold, pixel_threshold)
    summary_path = args.run_directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {"config": config}
    train_unique = dm.train.num_pos + dm.train.num_neg if hasattr(dm.train, "num_pos") else len(dm.train)
    summary["train_unique_size"] = train_unique
    summary["train_epoch_samples"] = len(dm.train)
    summary["val_size"], summary["test_size"] = len(dm.val), len(dm.test)
    if hasattr(dm.train, "num_pos"):
        summary["split_class_counts"] = {
            "train": {"normal": dm.train.num_neg, "defect": dm.train.num_pos},
            "validation": {"normal": dm.val.num_neg, "defect": dm.val.num_pos},
            "test": {"normal": dm.test.num_neg, "defect": dm.test.num_pos},
        }
    summary.pop("train_size", None)
    summary["metrics"] = metrics
    summary["metric_implementation"] = {"aupro": "anomalib 0.7 AUPRO, fpr_limit=0.3",
                                         "thresholds": "image Youden J and pixel Dice, validation only"}
    save_json(summary_path, summary)
    if not config["name"].startswith("mvtec") and not args.skip_panels:
        save_error_panels(test, image_threshold, pixel_threshold, args.run_directory / "error_cases")
    progress_writer(args.progress_file, "complete")(len(dm.test), len(dm.test))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
