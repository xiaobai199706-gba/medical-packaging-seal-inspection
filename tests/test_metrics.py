import numpy as np

from medical_anomaly.baseline import best_threshold, evaluate
from medical_anomaly.evaluation import (PixelHistogram, collect_array_pixel_histograms,
                                        evaluate_original_pixels, histogram_aupro,
                                        independent_image_metrics)


def test_validation_threshold_and_metrics_are_separate():
    threshold = best_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), "youden")
    masks = np.zeros((2, 4, 4), dtype=np.uint8); masks[1, 1:3, 1:3] = 1
    maps = np.full((2, 4, 4), 0.1); maps[1, 1:3, 1:3] = 0.9
    result = evaluate({"labels": np.array([0, 1]), "scores": np.array([0.1, 0.9]), "masks": masks, "maps": maps}, threshold, 0.5)
    assert result["image_recall"] == 1.0
    assert result["image_specificity"] == 1.0
    assert result["pixel_dice"] == 1.0


def test_independent_image_metrics_match_known_perfect_ranking():
    result = independent_image_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert result == {"auroc": 1.0, "ap": 1.0}


def test_histogram_pixel_metrics_are_perfect_for_separated_scores():
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[1, 2:6, 2:6] = 1
    maps = np.full((2, 8, 8), 0.1, dtype=np.float32)
    maps[1, 2:6, 2:6] = 0.9
    pixel = collect_array_pixel_histograms(masks, maps, bins=256)
    result = evaluate_original_pixels(pixel)
    assert result["pixel_auroc"] == 1.0
    assert result["pixel_ap"] == 1.0
    assert result["pixel_dice"] == 1.0
    assert result["pixel_aupro"] == 1.0


def test_aupro_fpr_uses_background_from_normal_images():
    # The defect region is detected, but one normal-image pixel has a higher
    # score. Including all mask-zero pixels therefore gives a non-zero FPR.
    masks = np.zeros((2, 2, 2), dtype=np.uint8)
    masks[1, 0, 0] = 1
    maps = np.zeros((2, 2, 2), dtype=np.float32)
    maps[0, 0, 0] = 0.95
    maps[1, 0, 0] = 0.9
    pixel = collect_array_pixel_histograms(masks, maps, bins=256)
    assert pixel.negative.sum() == 7
    assert 0 < histogram_aupro(pixel, max_fpr=0.3) < 1
