from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .noise import NoiseResult, inject_feature_independent_noise


CMW_MEAN = tuple(value / 255.0 for value in (125.3, 123.0, 113.9))
CMW_STD = tuple(value / 255.0 for value in (63.0, 62.1, 66.7))


class IndexedCIFAR(Dataset):
    def __init__(self, base: Dataset, observed_labels: Sequence[int]) -> None:
        if len(base) != len(observed_labels):
            raise ValueError("base dataset and observed_labels have different lengths")
        self.base = base
        self.observed_labels = np.asarray(observed_labels, dtype=np.int64)
        self.clean_labels = np.asarray(base.targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, _ = self.base[index]
        return (
            image,
            int(self.observed_labels[index]),
            int(self.clean_labels[index]),
            int(index),
        )


@dataclass
class CifarBundle:
    train: IndexedCIFAR
    eval_train: IndexedCIFAR
    test: IndexedCIFAR
    class_names: list[str]
    clean_labels: np.ndarray
    noisy_labels: np.ndarray
    true_transition: np.ndarray
    selected_mask: np.ndarray
    actual_noise_rate: float
    official_superclasses: list[list[int]] | None


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(CMW_MEAN, CMW_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, eval_transform


def _read_cifar100_superclasses(base: datasets.CIFAR100) -> list[list[int]]:
    raw_path = Path(base.root) / base.base_folder / base.train_list[0][0]
    with raw_path.open("rb") as handle:
        raw = pickle.load(handle, encoding="latin1")
    fine = np.asarray(raw["fine_labels"], dtype=np.int64)
    coarse = np.asarray(raw["coarse_labels"], dtype=np.int64)
    mapping = np.full(100, -1, dtype=np.int64)
    for fine_id in range(100):
        values = np.unique(coarse[fine == fine_id])
        if values.size != 1:
            raise RuntimeError(f"fine class {fine_id} has invalid coarse labels {values}")
        mapping[fine_id] = values[0]
    groups = [np.flatnonzero(mapping == coarse_id).tolist() for coarse_id in range(20)]
    if any(len(group) != 5 for group in groups):
        raise RuntimeError("CIFAR-100 did not yield 20 official superclasses of size 5")
    return groups


def load_cifar_bundle(
    *,
    dataset: str,
    root: str | Path,
    noise_type: str,
    rate: float,
    seed: int,
    download: bool,
    symmetric_include_self: bool = True,
) -> CifarBundle:
    name = dataset.lower().replace("-", "")
    train_transform, eval_transform = build_transforms()
    root = str(root)
    if name == "cifar10":
        dataset_class = datasets.CIFAR10
        num_classes = 10
    elif name == "cifar100":
        dataset_class = datasets.CIFAR100
        num_classes = 100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    train_base = dataset_class(root, train=True, transform=train_transform, download=download)
    eval_base = dataset_class(root, train=True, transform=eval_transform, download=False)
    test_base = dataset_class(root, train=False, transform=eval_transform, download=download)
    official_groups = (
        _read_cifar100_superclasses(eval_base) if name == "cifar100" else None
    )
    clean = np.asarray(train_base.targets, dtype=np.int64)
    noise: NoiseResult = inject_feature_independent_noise(
        clean,
        num_classes=num_classes,
        noise_type=noise_type,
        rate=rate,
        seed=seed,
        superclasses=official_groups,
        symmetric_include_self=symmetric_include_self,
    )
    test_labels = np.asarray(test_base.targets, dtype=np.int64)
    return CifarBundle(
        train=IndexedCIFAR(train_base, noise.noisy_labels),
        eval_train=IndexedCIFAR(eval_base, noise.noisy_labels),
        test=IndexedCIFAR(test_base, test_labels),
        class_names=list(train_base.classes),
        clean_labels=clean,
        noisy_labels=noise.noisy_labels,
        true_transition=noise.transition,
        selected_mask=noise.selected_mask,
        actual_noise_rate=noise.actual_rate,
        official_superclasses=official_groups,
    )
