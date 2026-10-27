from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_anomaly.baseline import aupro, best_threshold, load_upstream, predict, save_json, seed_all
from medical_anomaly.evaluation import (collect_array_pixel_histograms, collect_original_pixel_histograms, evaluate_original_pixels,
                                        image_metrics, independent_image_metrics, save_difficult_cases,
                                        save_score_diagnostics)


def progress_writer(path: Path | None, stage: str):
    def write(completed: int, total: int) -> None:
        if path is None:
            return
        payload = {"stage": stage, "completed": completed, "total": total, "updated_unix": time.time()}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    return write


def split_counts(dataset) -> dict[str, int]:
    return {"total": dataset.num_neg + dataset.num_pos, "normal": dataset.num_neg, "defect": dataset.num_pos}


def sample_paths(dataset) -> set[str]:
    normal, abnormal = dataset.samples
    return set(normal.image_path) | set(abnormal.image_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-archive", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--histogram-bins", type=int, default=65536)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--skip-panels", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); seed_all(config["seed"])
    if config.get("protocol") == "SSI-ContentGroup-Sparse3-v1":
        raise ValueError("Use run_content_group_experiment.py evaluate to freeze validation thresholds before test access")
    SuperSimpleNet, _, _ = load_upstream(args.upstream_root)
    from medical_anomaly.data.ds1_upstream import DS1UpstreamDataModule
    dm = DS1UpstreamDataModule(args.data_root, args.split_archive, tuple(config["image_size"]),
                               config["batch_size"], config["num_workers"], config["val_fraction"], config["seed"],
                               config.get("supervision_mode", "mixed"),
                               split_manifest=__import__('medical_anomaly.explicit_splits', fromlist=['manifest_argument']).manifest_argument(args, config),
                               manifest_sha256=config.get("split_manifest_sha256"))
    dm.setup()
    train_paths, val_paths, test_paths = sample_paths(dm.train), sample_paths(dm.val), sample_paths(dm.test)
    if train_paths & val_paths or train_paths & test_paths or val_paths & test_paths:
        raise RuntimeError("Train, validation, and test image paths must be mutually exclusive")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperSimpleNet(tuple(config["image_size"]), config).to(device)
    model.load_model(args.run_directory / "checkpoints" / "weights.pt")
    validation = predict(model, dm.val_dataloader(), device, progress_writer(args.progress_file, "validation_inference"))
    test = predict(model, dm.test_dataloader(), device, progress_writer(args.progress_file, "test_inference"))
    image_threshold = best_threshold(validation["labels"], validation["scores"], "youden")

    validation_pixel = collect_original_pixel_histograms(
        validation, args.histogram_bins, progress_writer(args.progress_file, "validation_original_scale")
    )
    validation_pixel_metrics = evaluate_original_pixels(validation_pixel)
    pixel_threshold = validation_pixel_metrics["pixel_threshold"]
    test_pixel = collect_original_pixel_histograms(
        test, args.histogram_bins, progress_writer(args.progress_file, "test_original_scale")
    )
    test_metrics = image_metrics(np.asarray(test["labels"]), np.asarray(test["scores"]), image_threshold)
    test_metrics.update(evaluate_original_pixels(test_pixel, pixel_threshold))

    reference_image = {"auroc": roc_auc_score(test["labels"], test["scores"]),
                       "ap": average_precision_score(test["labels"], test["scores"])}
    independent_image = independent_image_metrics(np.asarray(test["labels"]), np.asarray(test["scores"]))
    low_resolution_pixel = collect_array_pixel_histograms(test["masks"], test["maps"], args.histogram_bins)
    low_resolution_independent = evaluate_original_pixels(low_resolution_pixel, pixel_threshold)
    low_resolution_prediction = np.asarray(test["maps"]) >= pixel_threshold
    low_resolution_intersection = np.logical_and(low_resolution_prediction, test["masks"]).sum()
    low_resolution_reference = {
        "pixel_auroc": roc_auc_score(np.asarray(test["masks"]).ravel(), np.asarray(test["maps"]).ravel()),
        "pixel_ap": average_precision_score(np.asarray(test["masks"]).ravel(), np.asarray(test["maps"]).ravel()),
        "pixel_dice": float(2 * low_resolution_intersection / max(1, low_resolution_prediction.sum() + np.asarray(test["masks"]).sum())),
        "pixel_aupro": aupro(np.asarray(test["masks"]), np.asarray(test["maps"])),
    }
    diagnostics_dir = args.run_directory / "diagnostics"
    save_score_diagnostics(validation, test, image_threshold, diagnostics_dir)
    manifest = [] if args.skip_panels else save_difficult_cases(
        test, test_pixel, image_threshold, pixel_threshold, diagnostics_dir / "difficult_cases"
    )
    summary = {
        "config": config,
        "split_counts": {"train": split_counts(dm.train), "validation": split_counts(dm.val), "test": split_counts(dm.test),
                         "effective_train_epoch_samples": len(dm.train)},
        "data_isolation": {"pairwise_disjoint": True, "normalization": "fixed ImageNet constants; no fitted statistics",
                           "early_stopping": False, "checkpoint_selection": "final epoch only",
                           "threshold_selection": "validation only", "test_role": "final evaluation only"},
        "mask_resize": {"loader": "Albumentations mask target uses nearest-neighbor interpolation",
                        "evaluation": "heatmap bilinear-resized to 4110x3008; original nonzero-binarized mask is never resized"},
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "validation_pixel_metrics": validation_pixel_metrics,
        "test_metrics": test_metrics,
        "independent_cross_check": {
            "image_sklearn": reference_image,
            "image_independent_rank": independent_image,
            "image_absolute_difference": {key: abs(reference_image[key] - independent_image[key]) for key in reference_image},
            "pixel_256_reference_sklearn_anomalib": low_resolution_reference,
            "pixel_256_independent_histogram": {key: low_resolution_independent[key] for key in low_resolution_reference},
            "pixel_256_absolute_difference": {key: abs(low_resolution_reference[key] - low_resolution_independent[key]) for key in low_resolution_reference},
        },
        "aupro_definition": {"bins": args.histogram_bins, "fpr_limit": 0.3, "fpr_background": "all mask==0 pixels, including normal images",
                             "pro": "mean per connected ground-truth defect region", "integration": "linear endpoint interpolation; area / 0.3"},
        "difficult_case_counts": {key: sum(item["category"] == key for item in manifest) for key in sorted({item["category"] for item in manifest})},
    }
    save_json(args.run_directory / "diagnostics_summary.json", summary)
    progress_writer(args.progress_file, "complete")(len(test["labels"]), len(test["labels"]))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
