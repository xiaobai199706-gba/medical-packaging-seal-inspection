from __future__ import annotations

import hashlib
import math
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from medical_anomaly.baseline import best_threshold


LSE_TEMPERATURES = (0.05, 0.10, 0.25, 0.50, 1.00)


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    output = np.empty_like(values, dtype=np.float64)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def normalized_lse(maps: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    flat = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1)
    if flat.shape[1] == 0:
        raise ValueError("dense maps must contain at least one pixel")
    scaled = flat / temperature
    maximum = scaled.max(axis=1, keepdims=True)
    return temperature * (
        maximum[:, 0] + np.log(np.exp(scaled - maximum).mean(axis=1))
    )


def top_fraction_mean(maps: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    flat = np.asarray(maps).reshape(len(maps), -1)
    if flat.shape[1] == 0:
        raise ValueError("dense maps must contain at least one pixel")
    count = max(1, math.ceil(fraction * flat.shape[1]))
    boundary = flat.shape[1] - count
    return np.partition(flat, boundary, axis=1)[:, boundary:].mean(axis=1)


def aggregate_dense_scores(map_logits: np.ndarray) -> dict[str, np.ndarray]:
    logits = np.asarray(map_logits)
    if logits.ndim != 3:
        raise ValueError(f"expected [N,H,W] dense maps, got {logits.shape}")
    domains = {"logit": logits, "probability": stable_sigmoid(logits)}
    scores: dict[str, np.ndarray] = {}
    for domain, maps in domains.items():
        flat = maps.reshape(len(maps), -1)
        scores[f"map_{domain}_max"] = flat.max(axis=1)
        scores[f"map_{domain}_top_01"] = top_fraction_mean(maps, 0.01)
        scores[f"map_{domain}_top_05"] = top_fraction_mean(maps, 0.05)
        for temperature in LSE_TEMPERATURES:
            suffix = f"{temperature:.2f}".replace(".", "p")
            scores[f"map_{domain}_lse_{suffix}"] = normalized_lse(maps, temperature)
    return scores


def prediction_scores(score_logits: np.ndarray, map_logits: np.ndarray) -> dict[str, np.ndarray]:
    logits = np.asarray(score_logits, dtype=np.float64).reshape(-1)
    scores = {
        "head_logit": logits,
        "head_probability": stable_sigmoid(logits),
    }
    scores.update(aggregate_dense_scores(map_logits))
    return scores


def validate_alignment(
    sample_ids: np.ndarray,
    labels: np.ndarray,
    score_logits: np.ndarray,
    map_logits: np.ndarray,
) -> None:
    lengths = {len(sample_ids), len(labels), len(score_logits), len(map_logits)}
    if len(lengths) != 1:
        raise ValueError("sample IDs, labels, head scores, and dense maps are misaligned")
    if len(set(map(str, sample_ids))) != len(sample_ids):
        raise ValueError("sample IDs are not unique")
    if np.asarray(map_logits).ndim != 3:
        raise ValueError("dense maps must have shape [N,H,W]")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("image labels must be binary with normal=0 and defect=1")
    if not np.isfinite(score_logits).all() or not np.isfinite(map_logits).all():
        raise ValueError("predictions contain non-finite values")


def sample_id(image_path: str | Path, data_root: str | Path) -> str:
    path = Path(image_path).resolve()
    root = Path(data_root).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.as_posix()
    digest = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"{relative}#{digest}"


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = average_ranks(left), average_ranks(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "minimum": float(values.min()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "maximum": float(values.max()),
    }


def score_diagnostics(
    validation_labels: np.ndarray,
    validation_scores: np.ndarray,
    test_ids: np.ndarray,
    test_labels: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, object]:
    threshold = best_threshold(validation_labels, validation_scores, "youden")
    prediction = np.asarray(test_scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(test_labels, prediction, labels=[0, 1]).ravel()
    normal = np.asarray(test_labels) == 0
    defect = ~normal
    ids = np.asarray(test_ids).astype(str)
    normal_order = np.flatnonzero(normal)[np.argsort(test_scores[normal], kind="stable")[::-1]]
    defect_order = np.flatnonzero(defect)[np.argsort(test_scores[defect], kind="stable")]
    return {
        "threshold": float(threshold),
        "threshold_source": "validation_youden",
        "test": {
            "auroc": float(roc_auc_score(test_labels, test_scores)),
            "ap": float(average_precision_score(test_labels, test_scores)),
            "negated_score_auroc": float(roc_auc_score(test_labels, -np.asarray(test_scores))),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "normal_distribution": distribution_summary(test_scores[normal]),
            "defect_distribution": distribution_summary(test_scores[defect]),
            "false_positive_ids": ids[normal & prediction].tolist(),
            "false_negative_ids": ids[defect & ~prediction].tolist(),
            "highest_scoring_normal_ids": ids[normal_order[:10]].tolist(),
            "lowest_scoring_defect_ids": ids[defect_order[:10]].tolist(),
        },
    }


def error_overlap(reference: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key in ("false_positive_ids", "false_negative_ids"):
        left, right = set(reference[key]), set(candidate[key])
        union = left | right
        output[key.removesuffix("_ids")] = {
            "intersection_count": len(left & right),
            "union_count": len(union),
            "jaccard": float(len(left & right) / len(union)) if union else 1.0,
            "reference_only_ids": sorted(left - right),
            "candidate_only_ids": sorted(right - left),
        }
    return output


def rank_disagreements(
    sample_ids: np.ndarray,
    reference: np.ndarray,
    candidates: dict[str, np.ndarray],
    limit: int = 10,
) -> list[dict[str, object]]:
    reference_rank = average_ranks(reference) / max(1, len(reference) - 1)
    candidate_ranks = {
        name: average_ranks(score) / max(1, len(score) - 1)
        for name, score in candidates.items()
    }
    rows = []
    for index, identifier in enumerate(np.asarray(sample_ids).astype(str)):
        differences = {
            name: float(abs(reference_rank[index] - ranks[index]))
            for name, ranks in candidate_ranks.items()
        }
        name = max(differences, key=differences.get)
        rows.append({"sample_id": identifier, "max_rank_difference": differences[name],
                     "dense_score": name, "all_rank_differences": differences})
    return sorted(rows, key=lambda row: (-row["max_rank_difference"], row["sample_id"]))[:limit]


def restore_probability_map(map_logits: np.ndarray, source_size: tuple[int, int]) -> np.ndarray:
    height, width = source_size
    if height <= 0 or width <= 0:
        raise ValueError("source dimensions must be positive")
    probability = stable_sigmoid(np.asarray(map_logits)).astype(np.float32)
    return cv2.resize(probability, (width, height), interpolation=cv2.INTER_LINEAR)


def defect_morphology(image: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    grayscale = np.asarray(image)
    binary = np.asarray(mask) != 0
    if grayscale.ndim != 2 or binary.shape != grayscale.shape:
        raise ValueError("image and mask must be aligned 2D arrays")
    area = int(binary.sum())
    if area == 0:
        raise ValueError("defect morphology requires a non-empty mask")
    coordinates = np.column_stack(np.nonzero(binary))[:, ::-1].astype(np.float64)
    if len(coordinates) > 1:
        covariance = np.cov(coordinates, rowvar=False)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        elongation = math.sqrt(float(eigenvalues[-1])) / max(1.0, math.sqrt(float(eigenvalues[0])))
    else:
        elongation = 1.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1).astype(bool) & ~binary
    local_contrast = float(abs(grayscale[binary].mean() - grayscale[ring].mean())) if ring.any() else float("nan")

    low = cv2.resize(binary.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) != 0
    restored = cv2.resize(low.astype(np.uint8), (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_NEAREST) != 0
    union = np.logical_or(binary, restored).sum()
    visibility_iou = float(np.logical_and(binary, restored).sum() / union) if union else 1.0
    return {
        "area_pixels": area,
        "relative_area": float(area / binary.size),
        "pca_elongation": float(elongation),
        "local_contrast": local_contrast,
        "downsample_iou": visibility_iou,
        "downsample_foreground_pixels": int(low.sum()),
    }


def quartile_edges(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("quartile reference values must be finite and non-empty")
    return [float(value) for value in np.quantile(values, [0.25, 0.50, 0.75])]


def quartile_stratum(value: float, edges: list[float]) -> str:
    if len(edges) != 3:
        raise ValueError("quartile strata require three edges")
    # side='left' assigns an observation exactly on a boundary to the lower bin.
    return f"Q{int(np.searchsorted(np.asarray(edges), value, side='left')) + 1}"


def visibility_count_stratum(count: int) -> str:
    if count < 0:
        raise ValueError("foreground count cannot be negative")
    if count == 0:
        return "0"
    if count <= 4:
        return "1-4"
    if count <= 20:
        return "5-20"
    return ">20"
