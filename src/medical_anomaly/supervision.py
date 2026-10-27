from __future__ import annotations

from types import MethodType

import torch


SUPERVISION_MODES = {"mixed", "synthetic_only", "real_only"}


def supervision_mode(config: dict) -> str:
    mode = config.get("supervision_mode", "mixed")
    if mode not in SUPERVISION_MODES:
        raise ValueError(f"Unsupported supervision mode: {mode}")
    return mode


def _real_only_forward(self, features, adapted, mask, labels):
    """Keep upstream batch duplication but add no synthetic mask or noise."""
    adapted = torch.cat((adapted, adapted))
    mask = torch.cat((mask, mask))
    labels = torch.cat((labels, labels))
    duplicated_features = torch.cat((features, features)) if features is not None else None
    return duplicated_features, adapted, mask, labels


def configure_supervision(model, mode: str) -> None:
    if mode not in SUPERVISION_MODES:
        raise ValueError(f"Unsupported supervision mode: {mode}")
    if mode == "real_only":
        model.anomaly_generator.forward = MethodType(_real_only_forward, model.anomaly_generator)


def empty_exposure_counts(mode: str) -> dict[str, int | str]:
    return {
        "supervision_mode": mode,
        "input_samples": 0,
        "normal_image_exposures": 0,
        "real_defect_image_exposures": 0,
        "real_mask_positive_pixels": 0,
        "model_output_samples": 0,
        "synthetic_anomaly_exposures": 0,
        "synthetic_mask_positive_pixels": 0,
        "synthetic_noise_mask_positive_pixels": 0,
    }


def update_exposure_counts(counts: dict, labels, input_masks, output_masks) -> None:
    batch_size = int(labels.numel())
    duplicated_masks = torch.cat((input_masks, input_masks))
    synthetic_pixels = torch.logical_and(output_masks > 0, duplicated_masks == 0)
    counts["input_samples"] += batch_size
    counts["normal_image_exposures"] += int((labels == 0).sum().item())
    counts["real_defect_image_exposures"] += int((labels > 0).sum().item())
    counts["real_mask_positive_pixels"] += int((input_masks > 0).sum().item())
    counts["model_output_samples"] += int(output_masks.shape[0])
    counts["synthetic_anomaly_exposures"] += int(synthetic_pixels.flatten(1).any(dim=1).sum().item())
    counts["synthetic_mask_positive_pixels"] += int(synthetic_pixels.sum().item())
    counts["synthetic_noise_mask_positive_pixels"] += int(synthetic_pixels.sum().item())
