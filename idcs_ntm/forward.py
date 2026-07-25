from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelOutputs:
    probabilities: np.ndarray
    features: np.ndarray | None
    losses: np.ndarray
    observed_labels: np.ndarray


@torch.no_grad()
def collect_model_outputs(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    collect_features: bool,
) -> ModelOutputs:
    was_training = model.training
    model.eval()
    size = len(loader.dataset)
    probability: np.ndarray | None = None
    feature_array: np.ndarray | None = None
    losses = np.empty(size, dtype=np.float32)
    labels = np.empty(size, dtype=np.int64)
    for images, observed, _clean, indices in loader:
        images = images.to(device, non_blocking=True)
        observed_device = observed.to(device, non_blocking=True)
        if collect_features:
            logits, features = model(images, return_features=True)
        else:
            logits = model(images)
            features = None
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
        batch_losses = F.cross_entropy(logits, observed_device, reduction="none").cpu().numpy()
        index_array = indices.numpy()
        if probability is None:
            probability = np.empty((size, probabilities.shape[1]), dtype=np.float32)
            if features is not None:
                feature_array = np.empty(
                    (size, features.shape[1]), dtype=np.float32
                )
        probability[index_array] = probabilities
        losses[index_array] = batch_losses
        labels[index_array] = observed.numpy()
        if feature_array is not None and features is not None:
            feature_array[index_array] = features.cpu().numpy()
    model.train(was_training)
    if probability is None:
        raise ValueError("cannot collect outputs from an empty loader")
    return ModelOutputs(probability, feature_array, losses, labels)


def estimate_anchor_transition_from_probabilities(
    probabilities: np.ndarray,
    *,
    filter_outliers: bool,
    percentile: float = 97.0,
    probability_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Forward's T with one anchor point per clean class."""

    eta = np.asarray(probabilities, dtype=np.float64)
    if eta.ndim != 2 or eta.shape[0] == 0:
        raise ValueError("probabilities must have shape [N,C] with N > 0")
    num_classes = eta.shape[1]
    anchors = np.empty(num_classes, dtype=np.int64)
    transition = np.empty((num_classes, num_classes), dtype=np.float64)
    for class_id in range(num_classes):
        scores = eta[:, class_id]
        if filter_outliers:
            threshold = np.percentile(scores, percentile, method="higher")
            candidates = np.flatnonzero(scores < threshold)
            if not len(candidates):
                candidates = np.arange(len(scores))
            anchor = int(candidates[np.argmax(scores[candidates])])
        else:
            anchor = int(np.argmax(scores))
        anchors[class_id] = anchor
        transition[class_id] = eta[anchor]
    transition = np.maximum(transition, probability_floor)
    transition /= transition.sum(axis=1, keepdims=True)
    return transition.astype(np.float32), anchors


def estimate_anchor_transition(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    filter_outliers: bool,
) -> tuple[np.ndarray, np.ndarray, ModelOutputs]:
    outputs = collect_model_outputs(
        model, loader, device=device, collect_features=True
    )
    transition, anchors = estimate_anchor_transition_from_probabilities(
        outputs.probabilities, filter_outliers=filter_outliers
    )
    return transition, anchors, outputs
