from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from anomalib.data.utils import Split
from torch.utils.data import DataLoader

from datamodules.base import Supervision
from datamodules.base.datamodule import SSNDataModule
from datamodules.base.dataset import SSNDataset
from medical_anomaly.data.ds1 import _label, _portable_relative, _read_split, _stratified_indices


SUPERVISION_MODES = {"mixed", "synthetic_only", "real_only"}


class DS1UpstreamDataset(SSNDataset):
    """DS1 path adapter that delegates sampling and augmentation to upstream."""

    def __init__(self, rows, partition, data_root, transform, split, flips, seed=42,
                 supervision_mode="mixed"):
        if supervision_mode not in SUPERVISION_MODES:
            raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
        super().__init__(Path(data_root), Supervision.MIXED_SUPERVISION, transform, split,
                         flips=flips, normal_flips=False, dilate=7 if split == Split.TRAIN else None,
                         dt=(3, 2) if split == Split.TRAIN else None)
        self.rows, self.partition, self.seed = rows, partition, seed
        self.supervision_mode = supervision_mode

    def _setup(self):
        super()._setup()
        if self.split != Split.TRAIN:
            return
        capacity = max(self.num_neg, self.generated_num_pos)
        self._shared_counter = mp.Value("q", 0)
        self._shared_generation = mp.Value("q", 0)
        self._shared_active_length = mp.Value("q", 0)
        self._shared_frequency = mp.Array("d", self.num_neg, lock=True)
        self._shared_permutation = mp.Array("q", capacity, lock=True)
        self._initialize_permutation()

    def _frequency(self):
        return np.frombuffer(self._shared_frequency.get_obj(), dtype=np.float64)

    def _permutation(self):
        return np.frombuffer(self._shared_permutation.get_obj(), dtype=np.int64)

    def _choice(self, size, probability=None):
        generation = self._shared_generation.value
        rng = np.random.RandomState(self.seed + generation * 1_000_003)
        result = rng.choice(range(self.num_neg), size=size, p=probability, replace=self.replace)
        self._shared_generation.value = generation + 1
        return result

    def _initialize_permutation(self):
        size = self.generated_num_pos if self.replace else self.num_neg
        selected = self._choice(size)
        self._permutation()[:size] = selected
        self._shared_active_length.value = size
        self.neg_retrieval_freq = self._frequency()
        self.neg_imgs_permutation = self._permutation()[:size]
        self.counter = 0

    def generate_permutation(self):
        frequency = self._frequency()
        probability = 1 - (frequency / np.max(frequency))
        probability = probability - np.median(probability) + 1
        probability = probability ** (np.log(len(probability)) * 4)
        probability = probability / np.sum(probability)
        selected = self._choice(self.generated_num_pos, probability)
        self._permutation()[:self.generated_num_pos] = selected
        self._shared_active_length.value = self.generated_num_pos
        self._shared_counter.value = 0
        self.neg_retrieval_freq = frequency
        self.neg_imgs_permutation = self._permutation()[:self.generated_num_pos]
        self.counter = 0

    def checkpoint_state(self):
        return {
            "counter": self._shared_counter.value,
            "generation": self._shared_generation.value,
            "active_length": self._shared_active_length.value,
            "frequency": self._frequency().copy(),
            "permutation": self._permutation().copy(),
        }

    def load_checkpoint_state(self, state):
        self._shared_counter.value = state["counter"]
        self._shared_generation.value = state["generation"]
        self._shared_active_length.value = state["active_length"]
        self._frequency()[:] = state["frequency"]
        self._permutation()[:] = state["permutation"]

    def make_dataset(self):
        samples = []
        for row in self.rows:
            image_path = self.root / self.partition / _portable_relative(row["image_paths"])
            mask_value = (row.get("notation_paths") or "").strip()
            mask_path = self.root / _portable_relative(mask_value) if mask_value else Path("")
            if mask_value and not mask_path.exists():
                mask_path = self.root / "Notation" / mask_path.name
            samples.append({"image_path": str(image_path), "mask_path": str(mask_path) if mask_value else "",
                            "label_index": _label(row), "is_segmented": True})
        frame = pd.DataFrame(samples)
        return frame.loc[frame.label_index == 0].reset_index(drop=True), frame.loc[frame.label_index == 1].reset_index(drop=True)

    def get_sample_data(self, index):
        if self.split != Split.TRAIN or self.supervision_mode != "synthetic_only":
            return super().get_sample_data(index)

        # Preserve the official epoch length while replacing real defects with normals.
        permutation_index = index % self.generated_num_pos
        ix = self.neg_imgs_permutation[permutation_index]
        self.neg_retrieval_freq[ix] = self.neg_retrieval_freq[ix] + 1
        sample = self._normal_samples.iloc[ix]
        return sample.image_path, sample.mask_path, sample.label_index, sample.get("is_segmented", True)

    def __getitem__(self, index):
        if self.split == Split.TRAIN:
            self.counter = self._shared_counter.value
            self.neg_retrieval_freq = self._frequency()
            active = self._shared_active_length.value
            self.neg_imgs_permutation = self._permutation()[:active]
        item = super().__getitem__(index)
        if self.split == Split.TRAIN:
            self._shared_counter.value = self.counter
        item["mask"] = (item["mask"] != 0).to(torch.float32)
        return item


class DS1UpstreamDataModule(SSNDataModule):
    """Partial split DataModule with a train-only stratified validation carve-out."""

    def __init__(self, data_root: Path, split_archive: Path, image_size=(256, 256), batch_size=8,
                 num_workers=0, val_fraction=0.2, seed=42, supervision_mode="mixed",
                 split_manifest=None, manifest_sha256=None):
        if supervision_mode not in SUPERVISION_MODES:
            raise ValueError(f"Unsupported supervision mode: {supervision_mode}")
        super().__init__(data_root, Supervision.MIXED_SUPERVISION, image_size=image_size,
                         train_batch_size=batch_size, eval_batch_size=batch_size,
                         num_workers=num_workers, seed=seed, flips=True)
        self.supervision_mode = supervision_mode
        if split_manifest is not None:
            from medical_anomaly.explicit_splits import load_explicit_splits
            lists, self.explicit_manifest, self.manifest_sha256 = load_explicit_splits(
                split_manifest, data_root, manifest_sha256)
            train_rows, val_rows, test_rows = (lists[k] for k in ['train', 'validation', 'test'])
            train_partition = test_partition = ""
        else:
            rows = _read_split(split_archive, "TrainingXVal.csv")
            test_rows = _read_split(split_archive, "Testing.csv")
            train_idx, val_idx = _stratified_indices([_label(row) for row in rows], val_fraction, seed)
            train_rows, val_rows = [rows[i] for i in train_idx], [rows[i] for i in val_idx]
            train_partition, test_partition = "TrainingXVal", "Testing"
        self.train_data = DS1UpstreamDataset(train_rows, train_partition, data_root,
                                             self.transform_train, Split.TRAIN, flips=True, seed=seed,
                                             supervision_mode=supervision_mode)
        self.val_data = DS1UpstreamDataset(val_rows, train_partition, data_root,
                                           self.transform_eval, Split.VAL, flips=False)
        self.test_data = DS1UpstreamDataset(test_rows, test_partition, data_root, self.transform_eval, Split.TEST, flips=False)
        self.train_generator = torch.Generator().manual_seed(seed)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            generator=self.train_generator,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def checkpoint_state(self):
        return {
            "generator": self.train_generator.get_state(),
            "dataset": self.train_data.checkpoint_state(),
        }

    def load_checkpoint_state(self, state):
        self.train_generator.set_state(state["generator"])
        self.train_data.load_checkpoint_state(state["dataset"])

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)

    def _setup(self, _stage=None):
        self.train_data.setup(); self.val_data.setup(); self.test_data.setup()

    @property
    def train(self): return self.train_data
    @property
    def val(self): return self.val_data
    @property
    def test(self): return self.test_data
