"""Protocol-level regression checks for the published streaming evaluator."""
from pathlib import Path
import importlib.util
import sys

import cv2
import numpy as np
import pytest


@pytest.fixture
def evaluator(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    code = root / "tools" / "evidence_audit_20261023"
    monkeypatch.setenv("MEDICAL_ANOMALY_ROOT", str(root))
    monkeypatch.setenv("MEDICAL_ANOMALY_OUTPUT", str(tmp_path / "audit-output"))
    monkeypatch.syspath_prepend(str(code))
    for name in ("common_audit", "frozen_evaluator"):
        spec = importlib.util.spec_from_file_location(name, code / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
    return sys.modules["frozen_evaluator"]


def test_probability_preserves_absolute_score_domain(evaluator):
    probability = evaluator.stable_probability(np.array([-1000.0, 0.0, 1000.0]))
    assert probability.dtype == np.float32
    np.testing.assert_array_equal(probability, [0.0, 0.5, 1.0])
    # Values are absolute probabilities: one image's range cannot stretch another's.
    left = evaluator.stable_probability(np.array([-1.0, 1.0]))
    right = evaluator.stable_probability(np.array([-1.0, 1.0, 20.0]))
    np.testing.assert_array_equal(left, right[:2])


def test_streaming_metrics_include_normal_false_positive_pixels(evaluator, tmp_path, monkeypatch):
    height, width = 3008, 4110
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[100:108, 100:108] = 255
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(mask_path), mask)
    bundle = tmp_path / "two_samples.npz"
    np.savez_compressed(
        bundle,
        test_labels=np.array([1, 0], dtype=np.int8),
        test_map_logits=np.full((2, 256, 256), 2.0, dtype=np.float32),
        test_score_logits=np.array([2.0, -2.0], dtype=np.float32),
        test_mask_paths=np.array([str(mask_path), ""]),
        test_image_paths=np.array(["defect", "normal"]),
        test_source_sizes=np.array([[height, width], [height, width]], dtype=np.int32),
        test_sample_ids=np.array(["defect-id", "normal-id"]),
    )
    # Boundary implementation is separately covered by the existing spatial tests.
    monkeypatch.setattr(evaluator._mod, "normalized_boundary_distance", lambda *_: 0.0)
    result, per_image = evaluator.evaluate_bundle(bundle, 0.5, 0.5, progress=False)
    pixels = height * width
    # Constant scores have tied ranking. AP equals foreground prevalence across BOTH images.
    assert result["pixel_ap"] == pytest.approx(64 / (2 * pixels), abs=1e-15)
    assert result["pixel_dice"] == pytest.approx(128 / (2 * pixels + 64), abs=1e-15)
    assert result["iou"] == pytest.approx(64 / (2 * pixels), abs=1e-15)
    assert per_image[1]["fp"] == pixels
    assert result["samples"] == 2 and result["normal"] == result["defect"] == 1
    assert result["image_auroc"] == result["image_ap"] == 1.0
    assert result["image_threshold"] == result["pixel_threshold"] == 0.5
