import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_supervision_decomposition import normalized_boundary_distance


def test_identical_boundaries_have_zero_distance() -> None:
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 8:16] = True
    assert normalized_boundary_distance(mask, mask) == 0.0


def test_boundary_distance_is_symmetric() -> None:
    left = np.zeros((20, 30), dtype=bool)
    right = np.zeros((20, 30), dtype=bool)
    left[5:10, 8:16] = True
    right[5:10, 10:18] = True
    assert normalized_boundary_distance(left, right) == normalized_boundary_distance(right, left)
