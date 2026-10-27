from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


@dataclass
class PixelHistogram:
    positive: np.ndarray
    negative: np.ndarray
    regions: list[np.ndarray]
    per_image: list[dict[str, object]]
    bins: int


def _score_bins(values: np.ndarray, bins: int) -> np.ndarray:
    return np.minimum((np.clip(values, 0, 1) * (bins - 1)).astype(np.int32), bins - 1)


def _descending_counts(histogram: np.ndarray) -> np.ndarray:
    return np.cumsum(histogram[::-1], dtype=np.int64)


def histogram_binary_metrics(positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    tp, fp = _descending_counts(positive), _descending_counts(negative)
    positives, negatives = max(1, int(positive.sum())), max(1, int(negative.sum()))
    tpr, fpr = tp / positives, fp / negatives
    precision, recall = tp / np.maximum(1, tp + fp), tpr
    auroc = np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr])
    ap = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return {"auroc": float(auroc), "ap": float(ap)}


def histogram_dice_threshold(positive: np.ndarray, negative: np.ndarray) -> tuple[float, float]:
    tp, fp = _descending_counts(positive), _descending_counts(negative)
    dice = 2 * tp / np.maximum(1, int(positive.sum()) + tp + fp)
    descending_index = int(np.argmax(dice))
    bin_index = len(positive) - 1 - descending_index
    return float(bin_index / (len(positive) - 1)), float(dice[descending_index])


def histogram_aupro(pixel: PixelHistogram, max_fpr: float = 0.3) -> float:
    if not pixel.regions:
        return float("nan")
    fp = _descending_counts(pixel.negative).astype(np.float64)
    fpr = fp / max(1, int(pixel.negative.sum()))
    pro = np.mean([_descending_counts(region) / max(1, int(region.sum())) for region in pixel.regions], axis=0)
    # Curves run from the highest threshold to the lowest. Keep the FPR-limited
    # section and linearly add the endpoint when no score bin lands exactly on it.
    keep = fpr <= max_fpr
    x, y = np.r_[0.0, fpr[keep]], np.r_[0.0, pro[keep]]
    if x[-1] < max_fpr:
        right = int(np.flatnonzero(fpr > max_fpr)[0])
        left_x, left_y = x[-1], y[-1]
        right_x, right_y = fpr[right], pro[right]
        endpoint = left_y if right_x == left_x else left_y + (right_y - left_y) * (max_fpr - left_x) / (right_x - left_x)
        x, y = np.r_[x, max_fpr], np.r_[y, endpoint]
    return float(np.trapz(y, x) / max_fpr)


def collect_original_pixel_histograms(result: dict[str, object], bins: int = 65536, progress=None) -> PixelHistogram:
    positive = np.zeros(bins, dtype=np.int64)
    negative = np.zeros(bins, dtype=np.int64)
    regions: list[np.ndarray] = []
    per_image: list[dict[str, object]] = []
    for index, (low_map, label, image_path, mask_path) in enumerate(zip(
        result["maps"], result["labels"], result["paths"], result["mask_paths"]
    )):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Cannot decode image: {image_path}")
        height, width = image.shape
        heat = cv2.resize(np.asarray(low_map, dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
        if int(label) == 1:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise OSError(f"Cannot decode mask: {mask_path}")
            mask = mask != 0
        else:
            mask = np.zeros((height, width), dtype=bool)
        indexed = _score_bins(heat, bins)
        positive += np.bincount(indexed[mask], minlength=bins)
        negative += np.bincount(indexed[~mask], minlength=bins)
        components, component_map = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for component in range(1, components):
            regions.append(np.bincount(indexed[component_map == component], minlength=bins))
        per_image.append({"index": index, "image_path": str(image_path), "label": int(label),
                          "positive_hist": np.bincount(indexed[mask], minlength=bins),
                          "negative_hist": np.bincount(indexed[~mask], minlength=bins)})
        if progress is not None:
            progress(index + 1, len(result["labels"]))
    return PixelHistogram(positive, negative, regions, per_image, bins)


def collect_array_pixel_histograms(masks: np.ndarray, maps: np.ndarray, bins: int = 65536) -> PixelHistogram:
    positive = np.zeros(bins, dtype=np.int64)
    negative = np.zeros(bins, dtype=np.int64)
    regions: list[np.ndarray] = []
    per_image: list[dict[str, object]] = []
    for index, (mask, heat) in enumerate(zip(np.asarray(masks) != 0, np.asarray(maps))):
        indexed = _score_bins(heat, bins)
        pos_hist = np.bincount(indexed[mask], minlength=bins)
        neg_hist = np.bincount(indexed[~mask], minlength=bins)
        positive += pos_hist; negative += neg_hist
        components, component_map = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        for component in range(1, components):
            regions.append(np.bincount(indexed[component_map == component], minlength=bins))
        per_image.append({"index": index, "positive_hist": pos_hist, "negative_hist": neg_hist})
    return PixelHistogram(positive, negative, regions, per_image, bins)


def evaluate_original_pixels(pixel: PixelHistogram, threshold: float | None = None) -> dict[str, float]:
    metrics = histogram_binary_metrics(pixel.positive, pixel.negative)
    optimal_threshold, optimal_dice = histogram_dice_threshold(pixel.positive, pixel.negative)
    threshold = optimal_threshold if threshold is None else threshold
    threshold_bin = int(np.ceil(threshold * (pixel.bins - 1)))
    tp = int(pixel.positive[threshold_bin:].sum()); fp = int(pixel.negative[threshold_bin:].sum())
    fn = int(pixel.positive[:threshold_bin].sum())
    dice = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"pixel_auroc": metrics["auroc"], "pixel_ap": metrics["ap"], "pixel_dice": float(dice),
            "pixel_aupro": histogram_aupro(pixel), "pixel_threshold": float(threshold),
            "validation_optimal_dice": optimal_dice}


def independent_image_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_labels, sorted_scores = labels[order].astype(np.int64), scores[order]
    distinct = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    tp = np.cumsum(sorted_labels)[distinct]
    fp = (1 + distinct) - tp
    tpr, fpr = tp / max(1, labels.sum()), fp / max(1, len(labels) - labels.sum())
    precision = tp / np.maximum(1, tp + fp)
    recall = tpr
    return {"auroc": float(np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr])),
            "ap": float(np.sum(np.diff(np.r_[0.0, recall]) * precision))}


def image_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {"image_auroc": float(roc_auc_score(labels, scores)),
            "image_ap": float(average_precision_score(labels, scores)),
            "image_recall": float(tp / max(1, tp + fn)),
            "image_specificity": float(tn / max(1, tn + fp)),
            "false_positives": int(fp), "false_negatives": int(fn)}


def save_score_diagnostics(validation: dict[str, object], test: dict[str, object], threshold: float, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title in zip(axes, (validation, test), ("Validation", "Test")):
        labels, scores = np.asarray(data["labels"]), np.asarray(data["scores"])
        ax.hist(scores[labels == 0], bins=25, alpha=0.65, label="normal")
        ax.hist(scores[labels == 1], bins=25, alpha=0.65, label="defect")
        ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.4f}")
        ax.set(title=title, xlabel="image anomaly score", ylabel="count"); ax.legend()
    fig.tight_layout(); fig.savefig(output / "image_score_distributions.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for data, name in ((validation, "validation"), (test, "test")):
        labels, scores = np.asarray(data["labels"]), np.asarray(data["scores"])
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        axes[0].plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(labels, scores):.4f}")
        axes[1].plot(recall, precision, label=f"{name} AP={average_precision_score(labels, scores):.4f}")
        roc_i = int(np.argmin(np.abs(roc_thresholds - threshold)))
        pr_i = int(np.argmin(np.abs(pr_thresholds - threshold))) if len(pr_thresholds) else 0
        axes[0].scatter(fpr[roc_i], tpr[roc_i], s=45)
        if len(pr_thresholds): axes[1].scatter(recall[pr_i], precision[pr_i], s=45)
    axes[0].plot([0, 1], [0, 1], "k:"); axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC and validation threshold")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR and validation threshold")
    for ax in axes: ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(output / "image_roc_pr_curves.png", dpi=160); plt.close(fig)


def _image_dice(entry: dict[str, object], threshold: float, bins: int) -> float:
    threshold_bin = int(np.ceil(threshold * (bins - 1)))
    pos, neg = entry["positive_hist"], entry["negative_hist"]
    tp, fp, fn = int(pos[threshold_bin:].sum()), int(neg[threshold_bin:].sum()), int(pos[:threshold_bin].sum())
    return float(2 * tp / max(1, 2 * tp + fp + fn))


def save_difficult_cases(result: dict[str, object], pixel: PixelHistogram, image_threshold: float,
                         pixel_threshold: float, output: Path) -> list[dict[str, object]]:
    labels, scores = np.asarray(result["labels"]), np.asarray(result["scores"])
    normal = np.where(labels == 0)[0]; defect = np.where(labels == 1)[0]
    fp = normal[scores[normal] >= image_threshold]
    fn = defect[scores[defect] < image_threshold]
    categories = {
        "false_positive": fp,
        "false_negative": fn,
        "highest_scoring_normal": normal[np.argsort(scores[normal])[::-1][:5]],
        "lowest_scoring_defect": defect[np.argsort(scores[defect])[:5]],
        "lowest_pixel_dice_defect": np.array(sorted(defect, key=lambda i: _image_dice(pixel.per_image[i], pixel_threshold, pixel.bins))[:10]),
    }
    manifest = []
    for category, indices in categories.items():
        directory = output / category; directory.mkdir(parents=True, exist_ok=True)
        for rank, index in enumerate(indices, 1):
            image = cv2.cvtColor(cv2.imread(result["paths"][index]), cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            heat = cv2.resize(np.asarray(result["maps"][index], dtype=np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            if labels[index] == 1:
                mask = cv2.imread(result["mask_paths"][index], cv2.IMREAD_GRAYSCALE) != 0
            else:
                mask = np.zeros((height, width), dtype=bool)
            prediction = heat >= pixel_threshold
            display_size = (1024, round(1024 * height / width))
            panels = [cv2.resize(image, display_size, interpolation=cv2.INTER_AREA),
                      cv2.resize(mask.astype(np.uint8), display_size, interpolation=cv2.INTER_NEAREST),
                      cv2.resize(heat, display_size, interpolation=cv2.INTER_AREA),
                      cv2.resize(prediction.astype(np.uint8), display_size, interpolation=cv2.INTER_NEAREST)]
            fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
            for ax, panel, title in zip(axes, panels, ("image", "ground truth", "restored heatmap", "prediction")):
                ax.imshow(panel, cmap=None if title == "image" else "inferno"); ax.set_title(title); ax.axis("off")
            dice = _image_dice(pixel.per_image[index], pixel_threshold, pixel.bins) if labels[index] else None
            fig.suptitle(f"score={scores[index]:.5f}, image threshold={image_threshold:.5f}, pixel Dice={dice if dice is not None else 'N/A'}")
            fig.tight_layout(); filename = f"{rank:02d}_{Path(result['paths'][index]).stem}.jpg"
            fig.savefig(directory / filename, dpi=130); plt.close(fig)
            manifest.append({"category": category, "rank": rank, "image_path": result["paths"][index],
                             "score": float(scores[index]), "label": int(labels[index]), "pixel_dice": dice,
                             "is_image_error": bool(index in fp or index in fn), "file": str(Path(category) / filename)})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
