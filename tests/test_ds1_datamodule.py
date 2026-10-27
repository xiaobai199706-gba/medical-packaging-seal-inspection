import zipfile
from pathlib import Path

from medical_anomaly.data.ds1 import _label, _read_split, _stratified_indices


def test_class_vectors():
    assert _label({"classes": "[1 0]"}) == 0
    assert _label({"classes": "[0 1]"}) == 1


def test_tesnting_alias(tmp_path: Path):
    archive = tmp_path / "split.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("split/Tesnting.csv", "names,image_paths,notation_paths,classes\na,a,,[1 0]\n")
    assert len(_read_split(archive, "Testing.csv")) == 1


def test_stratified_split_is_disjoint():
    train, val = _stratified_indices([0] * 10 + [1] * 10, 0.2, 42)
    assert not set(train) & set(val)
    assert len(val) == 4
