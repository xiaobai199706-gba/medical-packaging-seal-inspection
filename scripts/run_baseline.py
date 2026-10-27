from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import time
from pathlib import Path

import torch
from anomalib.utils.metrics import AUROC, AUPRO
from torchmetrics import AveragePrecision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_anomaly.baseline import best_threshold, evaluate, load_upstream, predict, save_error_panels, save_json, seed_all
from medical_anomaly.checkpointing import git_state, load_checkpoint, restore_rng_state, sha256_file
from medical_anomaly.supervision import configure_supervision, supervision_mode
from medical_anomaly.training import train_with_epoch_checkpoints
from medical_anomaly.data import MVTecPillDataModule
from medical_anomaly.explicit_splits import manifest_argument


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-archive", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")); seed_all(config["seed"])
    mode = supervision_mode(config)
    SuperSimpleNet, focal_loss, _ = load_upstream(args.upstream_root)
    from train import test as upstream_test
    if config["name"].startswith("mvtec"):
        dm = MVTecPillDataModule(args.data_root, config["batch_size"], config["seed"], config["train_limit"], config["eval_limit"])
    else:
        from medical_anomaly.data.ds1_upstream import DS1UpstreamDataModule
        dm = DS1UpstreamDataModule(args.data_root, args.split_archive, tuple(config["image_size"]), config["batch_size"], config["num_workers"], config["val_fraction"], config["seed"], mode,
                                  split_manifest=manifest_argument(args, config), manifest_sha256=config.get("split_manifest_sha256"))
        if config.get("protocol") == "SSI-ContentGroup-Sparse3-v1":
            raise ValueError("Use run_content_group_experiment.py for frozen-update training/calibration; legacy evaluation is not the new protocol")
    dm.setup(); device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperSimpleNet(tuple(config["image_size"]), config).to(device)
    configure_supervision(model, mode)
    class ValidationProxy:
        def train_dataloader(self): return dm.train_dataloader()
        def test_dataloader(self): return dm.val_dataloader()
        def checkpoint_state(self): return dm.checkpoint_state()

    checkpoint_dir = args.output / "checkpoints"
    latest_checkpoint = checkpoint_dir / "latest.ckpt"
    if args.resume is None and latest_checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {latest_checkpoint}")
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "run.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Result directory is already locked: {lock_path}") from error
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    atexit.register(lock_path.unlink, missing_ok=True)
    repository_state = git_state(ROOT)
    run_metadata = {
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "config": config,
        "device": device,
        "checkpoint": str(latest_checkpoint),
        "output": str(args.output),
        "git": repository_state,
        "started_unix": time.time(),
        "resume": str(args.resume) if args.resume else None,
    }
    save_json(args.output / "run_metadata.json", run_metadata)
    optimizer, scheduler = model.get_optimizers()
    start_epoch = 0
    best_metric = None
    best_model_state = None
    epoch_losses = []
    supervision_exposure = None
    if args.resume:
        resumed = load_checkpoint(args.resume)
        if resumed["config"] != config:
            raise ValueError("Resume checkpoint configuration does not match the requested experiment")
        identity_keys = ("commit", "worktree_status", "diff_sha256", "source_tree_sha256")
        mismatched = [key for key in identity_keys if resumed["git"].get(key) != repository_state.get(key)]
        if mismatched:
            raise ValueError(f"Resume checkpoint code/worktree mismatch: {mismatched}")
        if resumed["completed_epoch"] >= config["epochs"]:
            raise ValueError("Resume checkpoint has already completed the configured training epochs")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        dm.load_checkpoint_state(resumed["dataloader_state"])
        restore_rng_state(resumed["rng_state"])
        start_epoch = resumed["completed_epoch"]
        best_metric = resumed["best_metric"]
        best_model_state = resumed["best_model_state"]
        epoch_losses = resumed.get("epoch_losses", [])
        supervision_exposure = resumed.get("supervision_exposure")

    started = time.time()
    validation_metrics, latest_info, supervision_exposure = train_with_epoch_checkpoints(
        model, config["epochs"], ValidationProxy(), device,
        {"I-AUROC": AUROC(), "AP-det": AveragePrecision(num_classes=1)},
        {"P-AUROC": AUROC(), "AUPRO": AUPRO(), "AP-loc": AveragePrecision(num_classes=1)},
        focal_loss, upstream_test, latest_checkpoint, config, repository_state,
        clip_grad=config["clip_grad"], start_epoch=start_epoch,
        optimizer=optimizer, scheduler=scheduler, best_metric=best_metric,
        best_model_state=best_model_state, epoch_losses=epoch_losses,
        supervision_exposure=supervision_exposure,
    )
    model.save_model(checkpoint_dir)
    reloaded = SuperSimpleNet(tuple(config["image_size"]), config).to(device)
    reloaded.load_model(checkpoint_dir / "weights.pt")
    val, test = predict(reloaded, dm.val_dataloader(), device), predict(reloaded, dm.test_dataloader(), device)
    image_threshold = best_threshold(val["labels"], val["scores"], "youden")
    pixel_threshold = best_threshold(val["masks"].ravel(), val["maps"].ravel(), "dice")
    metrics = evaluate(test, image_threshold, pixel_threshold)
    train_unique = dm.train.num_pos + dm.train.num_neg if hasattr(dm.train, "num_pos") else len(dm.train)
    summary = {"config": config, "device": device, "train_unique_size": train_unique,
               "train_epoch_samples": len(dm.train), "val_size": len(dm.val), "test_size": len(dm.test),
               "upstream_validation_metrics": validation_metrics,
               "supervision_exposure": supervision_exposure,
               "elapsed_seconds": time.time() - started, "metrics": metrics,
               "checkpoint": latest_info,
               "final_weights": {"path": str(checkpoint_dir / "weights.pt"),
                                 "bytes": (checkpoint_dir / "weights.pt").stat().st_size,
                                 "sha256": sha256_file(checkpoint_dir / "weights.pt")},
               "resume": {"source": str(args.resume) if args.resume else None, "start_epoch": start_epoch}}
    save_json(args.output / "summary.json", summary)
    if not config["name"].startswith("mvtec"):
        save_error_panels(test, image_threshold,pixel_threshold,args.output / "error_cases")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
