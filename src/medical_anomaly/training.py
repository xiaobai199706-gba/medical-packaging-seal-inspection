from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from medical_anomaly.checkpointing import atomic_save_checkpoint, capture_rng_state
from medical_anomaly.supervision import empty_exposure_counts, update_exposure_counts


def _cpu_state_dict(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((name, value.detach().cpu().clone()) for name, value in model.state_dict().items())


def train_with_epoch_checkpoints(
    model,
    epochs: int,
    datamodule,
    device: str,
    image_metrics,
    pixel_metrics,
    focal_loss,
    upstream_test: Callable[..., dict[str, float]],
    checkpoint_path: Path,
    config: dict[str, Any],
    git: dict[str, str],
    clip_grad: bool = True,
    threshold: float = 0.5,
    start_epoch: int = 0,
    optimizer=None,
    scheduler=None,
    best_metric: dict[str, Any] | None = None,
    best_model_state: dict[str, torch.Tensor] | None = None,
    epoch_losses: list[float] | None = None,
    supervision_exposure: dict[str, Any] | None = None,
    progress_state: dict[str, Any] | None = None,
    stop_after_epoch: int | None = None,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    """Run the pinned upstream loop with an epoch-boundary checkpoint callback."""
    model.to(device)
    if optimizer is None or scheduler is None:
        optimizer, scheduler = model.get_optimizers()
    train_loader = datamodule.train_dataloader()
    losses = list(epoch_losses or [])
    validation_metrics: dict[str, float] = {}
    exposure = supervision_exposure or empty_exposure_counts(config.get("supervision_mode", "mixed"))
    progress = progress_state if progress_state is not None else {
        "optimizer_steps": start_epoch * len(train_loader), "sample_exposure_counts": {},
        "epoch_order_sha256": [], "lr_events": []}
    update_schedule = config.get("scheduler_unit") == "optimizer_step"
    if config.get("optimizer_updates") is not None:
        if epochs * len(train_loader) != config["optimizer_updates"]:
            raise ValueError("Actual dataloader length does not produce the frozen update budget")
        if progress["optimizer_steps"] != start_epoch * len(train_loader):
            raise ValueError("Resume optimizer-step counter differs from epoch boundary")

    for epoch in range(start_epoch, min(epochs, stop_after_epoch or epochs)):
        model.train()
        total_loss = 0.0
        sample_order = []
        with tqdm(total=len(train_loader), desc=f"{epoch}/{epochs}", miniters=1, unit="batch") as prog_bar:
            for index, batch in enumerate(train_loader):
                optimizer.zero_grad()
                image_batch = batch["image"].to(device)
                mask = batch["mask"].type(torch.float32).to(device)
                mask = F.interpolate(mask.unsqueeze(1), size=(model.fh, model.fw), mode="bilinear", align_corners=True)
                mask = torch.where(mask < 0.5, torch.zeros_like(mask), torch.ones_like(mask))
                label = batch["label"].to(device).type(torch.float32)
                input_mask = mask.clone()
                input_label = label.clone()
                is_segmented = batch["is_segmented"].to(device).type(torch.float32)
                anomaly_map, score, mask, label = model.forward(image_batch, mask, label)
                update_exposure_counts(exposure, input_label, input_mask, mask)
                seg_focal = focal_loss(torch.sigmoid(anomaly_map), mask, reduction=None)
                seg_l1 = torch.zeros_like(anomaly_map)
                seg_l1[mask == 0] = torch.clip(anomaly_map[mask == 0] + threshold, min=0)
                seg_l1[mask > 0] = torch.clip(-anomaly_map[mask > 0] + threshold, min=0)
                if "loss_mask" in batch:
                    loss_mask = batch["loss_mask"].type(torch.float32).to(device)
                    loss_mask = F.interpolate(
                        loss_mask.unsqueeze(1), size=seg_focal.shape[-2:],
                        mode="bilinear", align_corners=True,
                    )
                    loss_mask = torch.cat((loss_mask, loss_mask))
                    seg_focal *= loss_mask
                    seg_l1 *= loss_mask
                is_segmented = torch.cat((is_segmented, is_segmented)).type(torch.bool)
                bad_loss = seg_l1[is_segmented][mask[is_segmented] > 0]
                good_loss = seg_l1[is_segmented][mask[is_segmented] == 0]
                focal_value = seg_focal[is_segmented]
                bad_loss = bad_loss.mean() if len(bad_loss) else 0
                good_loss = good_loss.mean() if len(good_loss) else 0
                focal_value = focal_value.mean() if len(focal_value) else 0
                loss = good_loss + bad_loss + focal_value + focal_loss(torch.sigmoid(score), label)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}, batch {index + 1}")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1) if clip_grad else None
                optimizer.step()
                progress["optimizer_steps"] += 1
                for path in batch.get("image_path", []):
                    sample_order.append(path)
                    counts = progress["sample_exposure_counts"]
                    counts[path] = counts.get(path, 0) + 1
                if update_schedule:
                    scheduler.step()
                    if progress["optimizer_steps"] in config["lr_milestones_updates"]:
                        event = {"after_optimizer_step": progress["optimizer_steps"],
                                 "learning_rates": scheduler.get_last_lr()}
                        progress["lr_events"].append(event)
                        print(f"LR_EVENT {event}", flush=True)
                total_loss += loss.detach().cpu().item()
                prog_bar.set_postfix(
                    batch_loss=np.round(loss.data.cpu().detach().numpy(), 5),
                    avg_loss=np.round(total_loss / (index + 1), 5),
                    norm=norm,
                )
                prog_bar.update(1)

            if epoch + 1 == epochs and not config.get("skip_training_validation", False):
                validation_metrics = upstream_test(
                    model=model,
                    datamodule=datamodule,
                    device=device,
                    image_metrics=image_metrics,
                    pixel_metrics=pixel_metrics,
                    normalize=True,
                )
        if not update_schedule:
            scheduler.step()
        progress["epoch_order_sha256"].append(hashlib.sha256("\n".join(sample_order).encode()).hexdigest())
        epoch_loss = float(total_loss / max(1, len(train_loader)))
        losses.append(epoch_loss)
        if config.get("final_weights_only", False):
            best_metric = {"name": "final_update_only", "value": epoch_loss, "epoch": epoch + 1}
            best_model_state = {}
        elif best_metric is None or epoch_loss < best_metric["value"]:
            best_metric = {
                "name": "training_epoch_loss",
                "mode": "min",
                "value": epoch_loss,
                "epoch": epoch + 1,
            }
            best_model_state = _cpu_state_dict(model)
        payload = {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "amp_grad_scaler": None,
            "amp_enabled": False,
            "completed_epoch": epoch + 1,
            "best_metric": best_metric,
            "best_model_state": best_model_state,
            "rng_state": capture_rng_state(),
            "dataloader_state": datamodule.checkpoint_state(),
            "config": config,
            "git": git,
            "epoch_losses": losses,
            "validation_metrics": validation_metrics,
            "supervision_exposure": exposure,
            "training_progress": progress,
        }
        if ((epoch + 1) % config.get("checkpoint_interval_epochs", 1) != 0
                and epoch + 1 not in [epochs, stop_after_epoch]):
            continue
        checkpoint_info = atomic_save_checkpoint(checkpoint_path, payload)
        print(
            f"CHECKPOINT epoch={epoch + 1} path={checkpoint_path} "
            f"sha256={checkpoint_info['sha256']}",
            flush=True,
        )

    return validation_metrics, checkpoint_info, exposure
