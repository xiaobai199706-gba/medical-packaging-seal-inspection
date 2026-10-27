import numpy as np
import pytest

from medical_anomaly.diagnostics import (aggregate_dense_scores, defect_morphology,
                                         normalized_lse, prediction_scores,
                                         quartile_stratum, rank_disagreements,
                                         restore_probability_map, spearman_correlation,
                                         top_fraction_mean, validate_alignment,
                                         visibility_count_stratum)


def test_top_fraction_uses_at_least_one_pixel_and_ceil_boundary():
    maps = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    np.testing.assert_array_equal(top_fraction_mean(maps, 0.01), [3, 7])
    np.testing.assert_array_equal(top_fraction_mean(maps, 0.5), [2.5, 6.5])


def test_lse_is_stable_for_large_logits_and_constant_maps():
    maps = np.full((2, 3, 4), 10_000.0, dtype=np.float64)
    result = normalized_lse(maps, 0.05)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, [10_000.0, 10_000.0])


def test_all_registered_aggregations_are_deterministic():
    maps = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    first = aggregate_dense_scores(maps)
    second = aggregate_dense_scores(maps.copy())
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_prediction_alignment_rejects_duplicates_and_length_mismatch():
    valid = (np.array(["a", "b"]), np.array([0, 1]), np.array([0.1, 0.9]), np.zeros((2, 2, 2)))
    validate_alignment(*valid)
    with pytest.raises(ValueError, match="not unique"):
        validate_alignment(np.array(["a", "a"]), *valid[1:])
    with pytest.raises(ValueError, match="misaligned"):
        validate_alignment(valid[0], valid[1], valid[2][:-1], valid[3])


def test_head_and_map_outputs_keep_one_row_per_sample():
    scores = prediction_scores(np.array([-1.0, 1.0]), np.zeros((2, 2, 2)))
    assert all(value.shape == (2,) for value in scores.values())
    assert scores["head_probability"][0] < scores["head_probability"][1]


def test_original_size_restoration_is_bilinear_and_has_requested_shape():
    restored = restore_probability_map(np.array([[0.0, 1.0], [2.0, 3.0]]), (5, 7))
    assert restored.shape == (5, 7)
    assert restored.dtype == np.float32
    assert np.isfinite(restored).all()


def test_spearman_ties_and_rank_disagreement_are_repeatable():
    reference = np.array([0.0, 1.0, 1.0, 3.0])
    assert spearman_correlation(reference, reference) == pytest.approx(1.0)
    candidates = {"map": reference[::-1]}
    first = rank_disagreements(np.array(["a", "b", "c", "d"]), reference, candidates, limit=2)
    second = rank_disagreements(np.array(["a", "b", "c", "d"]), reference, candidates, limit=2)
    assert first == second


def test_defect_morphology_and_frozen_stratum_boundaries():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[20:24, 10:50] = 100
    mask = image != 0
    result = defect_morphology(image, mask)
    assert result["area_pixels"] == 160
    assert result["relative_area"] == pytest.approx(160 / 4096)
    assert result["pca_elongation"] > 1
    assert result["downsample_foreground_pixels"] > 20
    assert quartile_stratum(1.0, [1.0, 2.0, 3.0]) == "Q1"
    assert quartile_stratum(1.1, [1.0, 2.0, 3.0]) == "Q2"
    assert [visibility_count_stratum(value) for value in (0, 1, 4, 5, 20, 21)] == [
        "0", "1-4", "1-4", "5-20", "5-20", ">20"
    ]
