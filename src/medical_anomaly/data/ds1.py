from __future__ import annotations

import csv
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


def _portable_relative(value: str) -> Path:
    return Path(*PureWindowsPath(value.strip()).parts)


def _read_split(split_archive: Path, filename: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(split_archive) as archive:
        matches = [name for name in archive.namelist() if Path(name).name.lower() == filename.lower()]
        if not matches and filename.lower() == "testing.csv":
            matches = [name for name in archive.namelist() if Path(name).name.lower() == "tesnting.csv"]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {filename} or Tesnting.csv in {split_archive}")
        text = archive.read(matches[0]).decode("utf-8-sig").replace("\r\r\n", "\n")
    return list(csv.DictReader(text.splitlines()))


def _label(row: dict[str, str]) -> int:
    classes = row["classes"].replace(" ", "")
    if classes == "[10]":
        return 0
    if classes == "[01]":
        return 1
    raise ValueError(f"Unknown class vector: {row['classes']}")


def _stratified_indices(labels: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    train, val = [], []
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        rng.shuffle(indices)
        count = max(1, round(len(indices) * val_fraction))
        val.extend(indices[:count])
        train.extend(indices[count:])
    return sorted(train), sorted(val)


class DS1Dataset(Dataset):
    """Resolve relative CSV paths only and convert every nonzero mask pixel to one."""

    def __init__(self, rows: list[dict[str, str]], data_root: Path, image_size: tuple[int, int], partition: str):
        self.rows, self.data_root, self.image_size, self.partition = rows, Path(data_root), image_size, partition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row, label = self.rows[index], _label(self.rows[index])
        image_path = self.data_root / self.partition / _portable_relative(row["image_paths"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"Cannot decode image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_value = (row.get("notation_paths") or "").strip()
        mask_path = self.data_root / _portable_relative(mask_value) if mask_value else None
        if mask_path is not None and not mask_path.exists():
            mask_path = self.data_root / "Notation" / Path(mask_path).name
        if mask_path is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            decoded = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                raise OSError(f"Cannot decode mask: {mask_path}")
            mask = (decoded != 0).astype(np.uint8)
        height, width = self.image_size
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        tensor = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return {"image": tensor, "mask": torch.from_numpy((mask != 0).astype(np.float32)),
                "label": torch.tensor(label), "is_segmented": torch.tensor(True),
                "image_path": str(image_path), "mask_path": str(mask_path or "")}


class BalancedTrainDataset(Dataset):
    """Match upstream mixed-supervision one-positive/one-normal epoch sampling."""

    def __init__(self, dataset: DS1Dataset, seed: int):
        self.dataset, self.seed = dataset, seed
        self.positive = [i for i, row in enumerate(dataset.rows) if _label(row) == 1]
        self.negative = [i for i, row in enumerate(dataset.rows) if _label(row) == 0]

    def __len__(self) -> int:
        return 2 * len(self.positive)

    def __getitem__(self, index: int) -> dict[str, object]:
        source = self.positive[index] if index < len(self.positive) else random.Random(self.seed + index).choice(self.negative)
        return self.dataset[source]


@dataclass
class DS1DataModule:
    data_root: Path
    split_archive: Path
    image_size: tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 0
    val_fraction: float = 0.2
    seed: int = 42

    def setup(self) -> None:
        rows = _read_split(self.split_archive, "TrainingXVal.csv")
        test_rows = _read_split(self.split_archive, "Testing.csv")
        train_idx, val_idx = _stratified_indices([_label(row) for row in rows], self.val_fraction, self.seed)
        self.train = DS1Dataset([rows[i] for i in train_idx], self.data_root, self.image_size, "TrainingXVal")
        self.train_balanced = BalancedTrainDataset(self.train, self.seed)
        self.val = DS1Dataset([rows[i] for i in val_idx], self.data_root, self.image_size, "TrainingXVal")
        self.test = DS1Dataset(test_rows, self.data_root, self.image_size, "Testing")

    def train_dataloader(self) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(self.train_balanced, batch_size=self.batch_size, shuffle=True, generator=generator,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
