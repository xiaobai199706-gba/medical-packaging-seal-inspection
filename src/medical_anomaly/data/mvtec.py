from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


class MVTecPillDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, Path | None, int]], image_size=(256, 256)):
        self.samples, self.image_size = samples, image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, label = self.samples[index]
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        mask = np.zeros(image.shape[:2], np.uint8) if mask_path is None else (cv2.imread(str(mask_path), 0) != 0).astype(np.uint8)
        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        image = TF.normalize(TF.to_tensor(image), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return {"image": image, "mask": torch.from_numpy(mask.astype(np.float32)), "label": torch.tensor(label),
                "is_segmented": torch.tensor(True), "image_path": str(image_path), "mask_path": str(mask_path or "")}


@dataclass
class MVTecPillDataModule:
    root: Path
    batch_size: int = 4
    seed: int = 42
    train_limit: int = 16
    eval_limit: int = 16

    def setup(self):
        category = Path(self.root) / "pill"
        train = [(p, None, 0) for p in sorted((category / "train" / "good").glob("*.png"))]
        test = []
        for defect_dir in sorted((category / "test").iterdir()):
            for image in sorted(defect_dir.glob("*.png")):
                if defect_dir.name == "good":
                    test.append((image, None, 0))
                else:
                    mask = category / "ground_truth" / defect_dir.name / f"{image.stem}_mask.png"
                    test.append((image, mask, 1))
        rng = random.Random(self.seed); rng.shuffle(train)
        normal, abnormal = [x for x in test if x[2] == 0], [x for x in test if x[2] == 1]
        rng.shuffle(normal); rng.shuffle(abnormal)
        half = max(2, self.eval_limit // 2)
        self.train = MVTecPillDataset(train[:self.train_limit])
        self.val = MVTecPillDataset(normal[:half] + abnormal[:half])
        self.test = MVTecPillDataset(normal[half:2*half] + abnormal[half:2*half])

    def train_dataloader(self): return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=0)
    def val_dataloader(self): return DataLoader(self.val, batch_size=self.batch_size, shuffle=False, num_workers=0)
    def test_dataloader(self): return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=0)
