from __future__ import annotations

import argparse

import numpy as np
import torch

from idcs_ntm.models import IDCSTransition, cifar_resnet34, forward_corrected_nll
from idcs_ntm.utils import resolve_device


def _groups(num_classes: int) -> list[list[int]]:
    if num_classes == 10:
        return [list(range(10))]
    return [list(range(start, start + 5)) for start in range(0, 100, 5)]


def _prior(num_classes: int, groups: list[list[int]]) -> np.ndarray:
    transition = np.zeros((num_classes, num_classes), dtype=np.float32)
    for group in groups:
        off_diagonal = 0.2 / (len(group) - 1)
        for source in group:
            transition[source, group] = off_diagonal
            transition[source, source] = 0.8
    return transition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One synthetic forward/backward pass without downloading CIFAR"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, choices=(10, 100), default=100)
    options = parser.parse_args()

    device = resolve_device(options.device)
    groups = _groups(options.num_classes)
    classifier = cifar_resnet34(options.num_classes).to(device)
    transition_model = IDCSTransition(
        feature_dim=classifier.feature_dim,
        hidden_dim=100,
        num_classes=options.num_classes,
        superclasses=groups,
        initial_transition=_prior(options.num_classes, groups),
    ).to(device)

    images = torch.randn(options.batch_size, 3, 32, 32, device=device)
    observed = torch.randint(
        options.num_classes, (options.batch_size,), device=device
    )
    logits, features = classifier(images, return_features=True)
    matrices = transition_model(features)
    loss = forward_corrected_nll(logits, observed, matrices)
    loss.backward()

    print(
        f"ok device={device} logits={tuple(logits.shape)} "
        f"transition={tuple(matrices.shape)} loss={loss.item():.6f}"
    )
    if device.type == "cuda":
        print(f"peak_cuda_memory_mib={torch.cuda.max_memory_allocated(device)/(1024**2):.1f}")


if __name__ == "__main__":
    main()
