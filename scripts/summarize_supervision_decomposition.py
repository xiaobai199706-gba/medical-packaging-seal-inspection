from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTINUOUS_METRICS = (
    "image_auroc",
    "image_ap",
    "image_recall",
    "image_specificity",
    "pixel_auroc",
    "pixel_ap",
    "dice",
    "iou",
    "defect_region_recall",
    "normalized_boundary_distance",
    "pixel_aupro",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contour(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel).astype(bool)


def normalized_boundary_distance(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt_boundary = contour(ground_truth)
    pred_boundary = contour(prediction)
    if not gt_boundary.any() or not pred_boundary.any():
        return float("nan")
    distance_to_gt = cv2.distanceTransform(
        (~gt_boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5
    )
    distance_to_pred = cv2.distanceTransform(
        (~pred_boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_5
    )
    symmetric = 0.5 * (
        float(distance_to_pred[gt_boundary].mean())
        + float(distance_to_gt[pred_boundary].mean())
    )
    return symmetric / math.hypot(*ground_truth.shape)


def load_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise OSError(f"cannot decode mask: {path}")
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match source shape {shape}: {path}")
    return mask != 0


def restore_probability(logits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    probability = np.empty_like(values)
    positive = values >= 0
    probability[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    probability[~positive] = exponent / (1.0 + exponent)
    return cv2.resize(
        probability.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR
    )


def evaluate_spatial(bundle_path: Path, threshold: float) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    totals = {"tp": 0, "fp": 0, "fn": 0}
    boundary_distances: list[float] = []
    disappearances = 0
    rows: list[dict[str, object]] = []
    with np.load(bundle_path, allow_pickle=False) as bundle:
        labels = bundle["test_labels"]
        logits = bundle["test_map_logits"]
        masks = bundle["test_mask_paths"].astype(str)
        sizes = bundle["test_source_sizes"]
        sample_ids = bundle["test_sample_ids"].astype(str)
        if len(labels) != 360 or int((labels == 0).sum()) != 180 or int((labels == 1).sum()) != 180:
            raise ValueError(f"unexpected fixed-test composition in {bundle_path}")
        for index, (label, map_logits, mask_path, size, sample_id) in enumerate(
            zip(labels, logits, masks, sizes, sample_ids)
        ):
            shape = tuple(map(int, size))
            prediction = restore_probability(map_logits, shape) >= threshold
            if int(label) == 1:
                ground_truth = load_mask(mask_path, shape)
            else:
                ground_truth = np.zeros(shape, dtype=bool)
            tp = int(np.logical_and(prediction, ground_truth).sum())
            fp = int(np.logical_and(prediction, ~ground_truth).sum())
            fn = int(np.logical_and(~prediction, ground_truth).sum())
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
            distance = float("nan")
            disappeared = 0
            recall = float("nan")
            if int(label) == 1:
                recall = tp / (tp + fn)
                disappeared = int(tp == 0)
                disappearances += disappeared
                distance = normalized_boundary_distance(ground_truth, prediction)
                if math.isfinite(distance):
                    boundary_distances.append(distance)
            rows.append(
                {
                    "index": index,
                    "sample_id": sample_id,
                    "label": int(label),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "defect_region_recall": recall,
                    "normalized_boundary_distance": distance,
                    "complete_disappearance": disappeared,
                }
            )
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    return (
        {
            "iou": tp / (tp + fp + fn),
            "defect_region_recall": tp / (tp + fn),
            "normalized_boundary_distance": float(np.mean(boundary_distances)),
            "normalized_boundary_distance_n": len(boundary_distances),
            "complete_disappearance_count": disappearances,
        },
        rows,
    )


def validate_mixed_reference(rows: list[dict[str, object]], reference: dict[int, dict[str, str]]) -> None:
    for row in rows:
        expected = reference[int(row["index"])]
        for key in ("tp", "fp", "fn", "complete_disappearance"):
            tolerance = 1 if key in {"tp", "fp", "fn"} else 0
            if abs(int(row[key]) - int(expected[key])) > tolerance:
                raise AssertionError(
                    f"mixed regression mismatch at index {row['index']} for {key}: "
                    f"{row[key]} != {expected[key]}"
                )
        for key in ("defect_region_recall", "normalized_boundary_distance"):
            actual = float(row[key])
            reference_key = "defect_recall" if key == "defect_region_recall" else key
            wanted = float(expected[reference_key]) if expected[reference_key] else float("nan")
            if math.isnan(actual) and math.isnan(wanted):
                continue
            if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=2e-5):
                raise AssertionError(
                    f"mixed regression mismatch at index {row['index']} for {key}: {actual} != {wanted}"
                )


def spatial_from_reference(reference: dict[int, dict[str, str]]) -> dict[str, float | int]:
    rows = list(reference.values())
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    distances = [float(row["normalized_boundary_distance"]) for row in rows if row["normalized_boundary_distance"]]
    return {
        "iou": tp / (tp + fp + fn),
        "defect_region_recall": tp / (tp + fn),
        "normalized_boundary_distance": float(np.mean(distances)),
        "normalized_boundary_distance_n": len(distances),
        "complete_disappearance_count": sum(int(row["complete_disappearance"]) for row in rows),
    }


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--mixed-reference-csv", type=Path, required=True)
    args = parser.parse_args()
    entries = json.loads(args.index.read_text(encoding="utf-8"))["runs"]
    args.output_directory.mkdir(parents=True, exist_ok=True)

    reference: dict[str, dict[int, dict[str, str]]] = {}
    with args.mixed_reference_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            reference.setdefault(row["run"], {})[int(row["index"])] = row

    run_rows: list[dict[str, object]] = []
    per_sample_rows: list[dict[str, object]] = []
    file_manifest: list[dict[str, object]] = []
    test_order_hash: str | None = None
    for entry in entries:
        print(f"[start] {entry['run_id']}", flush=True)
        bundle_path = ROOT / entry["prediction_bundle"]
        manifest_path = ROOT / entry["prediction_manifest"]
        diagnostics_path = ROOT / entry["diagnostics_summary"]
        config_path = ROOT / entry["config"]
        run_directory = ROOT / entry["run_directory"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if sha256_file(bundle_path) != manifest["bundle"]["sha256"]:
            raise ValueError(f"prediction hash mismatch: {bundle_path}")
        with np.load(bundle_path, allow_pickle=False) as bundle:
            order_hash = hashlib.sha256("\n".join(bundle["test_sample_ids"].astype(str)).encode()).hexdigest()
        if test_order_hash is None:
            test_order_hash = order_hash
        elif order_hash != test_order_hash:
            raise ValueError(f"fixed test order differs: {entry['run_id']}")
        spatial, sample_rows = evaluate_spatial(bundle_path, diagnostics["pixel_threshold"])
        if entry["mode"] == "mixed":
            validate_mixed_reference(sample_rows, reference[entry["run_id"]])
            spatial = spatial_from_reference(reference[entry["run_id"]])
        metrics = diagnostics["test_metrics"]
        row = {
            "run_id": entry["run_id"],
            "mode": entry["mode"],
            "seed": entry["seed"],
            "optimizer_updates": 1020,
            "epochs": diagnostics["config"]["epochs"],
            "image_threshold": diagnostics["image_threshold"],
            "pixel_threshold": diagnostics["pixel_threshold"],
            "image_auroc": metrics["image_auroc"],
            "image_ap": metrics["image_ap"],
            "image_recall": metrics["image_recall"],
            "image_specificity": metrics["image_specificity"],
            "false_positives": metrics["false_positives"],
            "false_negatives": metrics["false_negatives"],
            "pixel_auroc": metrics["pixel_auroc"],
            "pixel_ap": metrics["pixel_ap"],
            "dice": metrics["pixel_dice"],
            "pixel_aupro": metrics["pixel_aupro"],
            **spatial,
        }
        run_rows.append(row)
        per_sample_rows.extend({"run_id": entry["run_id"], "mode": entry["mode"], "seed": entry["seed"], **r} for r in sample_rows)
        for kind, path in (
            ("config", config_path),
            ("diagnostics_summary", diagnostics_path),
            ("checkpoint", run_directory / "checkpoints" / "latest.ckpt"),
            ("weights", run_directory / "checkpoints" / "weights.pt"),
            ("prediction_bundle", bundle_path),
            ("prediction_manifest", manifest_path),
        ):
            file_manifest.append(
                {"run_id": entry["run_id"], "kind": kind, "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        print(
            f"[complete] {entry['run_id']} pixel_ap={row['pixel_ap']:.6f} "
            f"iou={row['iou']:.6f} disappearances={row['complete_disappearance_count']}",
            flush=True,
        )

    by_mode = {mode: [row for row in run_rows if row["mode"] == mode] for mode in ("mixed", "synthetic_only", "real_only")}
    aggregates: dict[str, object] = {}
    for mode, rows in by_mode.items():
        aggregates[mode] = {metric: stats([float(row[metric]) for row in rows]) for metric in CONTINUOUS_METRICS}
        aggregates[mode]["complete_disappearance_count"] = {
            "per_seed": [int(row["complete_disappearance_count"]) for row in rows],
            "total": sum(int(row["complete_disappearance_count"]) for row in rows),
        }

    deltas = []
    mixed_by_seed = {int(row["seed"]): row for row in by_mode["mixed"]}
    for mode in ("synthetic_only", "real_only"):
        for row in by_mode[mode]:
            base = mixed_by_seed[int(row["seed"])]
            deltas.append(
                {
                    "mode": mode,
                    "seed": row["seed"],
                    **{f"delta_{metric}": float(row[metric]) - float(base[metric]) for metric in CONTINUOUS_METRICS},
                    "delta_complete_disappearance_count": int(row["complete_disappearance_count"]) - int(base["complete_disappearance_count"]),
                }
            )
    pixel_ap_direction = {
        mode: [next(d for d in deltas if d["mode"] == mode and int(d["seed"]) == int(row["seed"]))["delta_pixel_ap"] for row in by_mode[mode]]
        for mode in ("synthetic_only", "real_only")
    }
    mixed_mean = aggregates["mixed"]["pixel_ap"]["mean"]
    syn_gap = mixed_mean - aggregates["synthetic_only"]["pixel_ap"]["mean"]
    real_gap = mixed_mean - aggregates["real_only"]["pixel_ap"]["mean"]
    support_reversals = {
        mode: {
            "dice_mean_delta_vs_mixed": aggregates[mode]["dice"]["mean"] - aggregates["mixed"]["dice"]["mean"],
            "iou_mean_delta_vs_mixed": aggregates[mode]["iou"]["mean"] - aggregates["mixed"]["iou"]["mean"],
            "boundary_distance_mean_delta_vs_mixed": aggregates[mode]["normalized_boundary_distance"]["mean"] - aggregates["mixed"]["normalized_boundary_distance"]["mean"],
        }
        for mode in ("synthetic_only", "real_only")
    }
    no_support_reversal = all(
        values["dice_mean_delta_vs_mixed"] <= 0
        and values["iou_mean_delta_vs_mixed"] <= 0
        and values["boundary_distance_mean_delta_vs_mixed"] >= 0
        for values in support_reversals.values()
    )
    decision = {
        "practical_pixel_ap_threshold": 0.01,
        "mixed_minus_synthetic_only_mean_pixel_ap": syn_gap,
        "mixed_minus_real_only_mean_pixel_ap": real_gap,
        "synthetic_only_same_seed_deltas": pixel_ap_direction["synthetic_only"],
        "real_only_same_seed_deltas": pixel_ap_direction["real_only"],
        "support_metric_deltas_vs_mixed": support_reversals,
        "classification": (
            "stable_complementarity"
            if syn_gap >= 0.01 and real_gap >= 0.01
            and all(value <= -0.01 for value in pixel_ap_direction["synthetic_only"])
            and all(value <= -0.01 for value in pixel_ap_direction["real_only"])
            and no_support_reversal
            else "insufficient_or_conflicting_mechanism_evidence"
        ),
    }
    output = {
        "schema_version": 1,
        "scope": "P01 1020-update supervision-source decomposition; descriptive three-seed evidence",
        "test_order_sha256": test_order_hash,
        "boundary_definition": "3x3 morphological-gradient contours; bidirectional symmetric mean OpenCV L2/DIST_MASK_5 distance / hypot(3008,4110)",
        "std_definition": "sample standard deviation, ddof=1",
        "runs": run_rows,
        "aggregates": aggregates,
        "same_seed_deltas_vs_mixed": deltas,
        "pre_registered_decision": decision,
    }
    (args.output_directory / "supervision_decomposition_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name, rows in (("mixed_vs_syn_vs_real.csv", run_rows), ("same_seed_deltas_vs_mixed.csv", deltas), ("per_sample_spatial_metrics.csv", per_sample_rows), ("artifact_sha256_manifest.csv", file_manifest)):
        with (args.output_directory / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"status": "complete", "runs": len(run_rows), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
