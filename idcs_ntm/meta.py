from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.func import functional_call
from torch.utils.data import Dataset

from .models.transition import IDCSTransition, forward_corrected_nll


def select_balanced_low_loss_indices(
    losses: np.ndarray,
    observed_labels: np.ndarray,
    *,
    num_classes: int,
    per_class: int,
) -> np.ndarray:
    losses = np.asarray(losses)
    labels = np.asarray(observed_labels)
    selected: list[np.ndarray] = []
    for class_id in range(num_classes):
        candidates = np.flatnonzero(labels == class_id)
        if not len(candidates):
            raise ValueError(f"observed class {class_id} has no candidate meta samples")
        count = min(per_class, len(candidates))
        order = np.argsort(losses[candidates], kind="stable")[:count]
        selected.append(candidates[order])
    return np.concatenate(selected).astype(np.int64)


class MetaSubset(Dataset):
    def __init__(
        self,
        indexed_dataset: Dataset,
        indices: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.indexed_dataset = indexed_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.int64)
        if self.indices.shape != self.targets.shape:
            raise ValueError("indices and targets must have the same shape")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source_index = int(self.indices[item])
        image, _observed, _clean, _ = self.indexed_dataset[source_index]
        return image, int(self.targets[item]), source_index


def cycle_loader(loader) -> Iterator:
    while True:
        yield from loader


@contextmanager
def frozen_parameters(module: nn.Module):
    parameters = list(module.parameters())
    previous = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, requires_grad in zip(parameters, previous):
            parameter.requires_grad_(requires_grad)


def mixup_meta(
    images: Tensor, targets: Tensor, *, alpha: float
) -> tuple[Tensor, Tensor, Tensor, float]:
    if alpha <= 0:
        return images, targets, targets, 1.0
    distribution = torch.distributions.Beta(alpha, alpha)
    coefficient = float(distribution.sample(()).item())
    coefficient = max(coefficient, 1.0 - coefficient)
    permutation = torch.randperm(images.shape[0], device=images.device)
    mixed = coefficient * images + (1.0 - coefficient) * images[permutation]
    return mixed, targets, targets[permutation], coefficient


def _functional_classifier(
    classifier: nn.Module,
    parameters: OrderedDict[str, Tensor],
    images: Tensor,
    *,
    return_features: bool,
):
    buffers = OrderedDict(
        (name, value.detach().clone()) for name, value in classifier.named_buffers()
    )
    return functional_call(
        classifier,
        (parameters, buffers),
        (images,),
        kwargs={"return_features": return_features},
        strict=True,
    )


def _idcs_loss_functional(
    classifier: nn.Module,
    parameters: OrderedDict[str, Tensor],
    transition_model: IDCSTransition,
    images: Tensor,
    observed_targets: Tensor,
) -> Tensor:
    logits, features = _functional_classifier(
        classifier, parameters, images, return_features=True
    )
    transition = transition_model(features)
    return forward_corrected_nll(logits, observed_targets, transition)


@dataclass(frozen=True)
class MetaStepDiagnostics:
    meta_loss: float
    finite_difference_epsilon: float
    delta_norm: float
    train_loss_plus: float
    train_loss_minus: float


def finite_difference_meta_step(
    *,
    classifier: nn.Module,
    transition_model: IDCSTransition,
    transition_optimizer: torch.optim.Optimizer,
    train_images: Tensor,
    train_targets: Tensor,
    meta_images: Tensor,
    meta_targets: Tensor,
    inner_learning_rate: float,
    finite_difference_scale: float = 0.01,
    mixup_alpha: float = 1.0,
) -> MetaStepDiagnostics:
    """Paper equations (6-8)--(6-11) using a finite-difference HVP."""

    parameters = OrderedDict(classifier.named_parameters())
    inner_loss = _idcs_loss_functional(
        classifier, parameters, transition_model, train_images, train_targets
    )
    inner_gradients = torch.autograd.grad(
        inner_loss, tuple(parameters.values()), create_graph=False
    )
    virtual = OrderedDict(
        (
            name,
            (parameter - inner_learning_rate * gradient)
            .detach()
            .requires_grad_(True),
        )
        for (name, parameter), gradient in zip(parameters.items(), inner_gradients)
    )

    mixed_images, first_targets, second_targets, coefficient = mixup_meta(
        meta_images, meta_targets, alpha=mixup_alpha
    )
    meta_logits = _functional_classifier(
        classifier, virtual, mixed_images, return_features=False
    )
    meta_loss = coefficient * F.cross_entropy(meta_logits, first_targets)
    meta_loss = meta_loss + (1.0 - coefficient) * F.cross_entropy(
        meta_logits, second_targets
    )
    delta = torch.autograd.grad(meta_loss, tuple(virtual.values()), allow_unused=False)
    delta_norm_tensor = torch.sqrt(
        sum(torch.sum(component.detach() ** 2) for component in delta)
    )
    delta_norm = float(delta_norm_tensor.item())
    epsilon = finite_difference_scale / max(delta_norm, 1e-12)

    plus = OrderedDict(
        (name, (parameter.detach() + epsilon * direction.detach()))
        for (name, parameter), direction in zip(parameters.items(), delta)
    )
    minus = OrderedDict(
        (name, (parameter.detach() - epsilon * direction.detach()))
        for (name, parameter), direction in zip(parameters.items(), delta)
    )
    loss_plus = _idcs_loss_functional(
        classifier, plus, transition_model, train_images, train_targets
    )
    theta_parameters = tuple(transition_model.parameters())
    gradient_plus = torch.autograd.grad(loss_plus, theta_parameters)
    loss_minus = _idcs_loss_functional(
        classifier, minus, transition_model, train_images, train_targets
    )
    gradient_minus = torch.autograd.grad(loss_minus, theta_parameters)

    coefficient_fd = -inner_learning_rate / (2.0 * epsilon)
    transition_optimizer.zero_grad(set_to_none=True)
    for parameter, plus_gradient, minus_gradient in zip(
        theta_parameters, gradient_plus, gradient_minus
    ):
        parameter.grad = coefficient_fd * (plus_gradient - minus_gradient).detach()
    transition_optimizer.step()
    transition_model.enforce_mask_()
    return MetaStepDiagnostics(
        meta_loss=float(meta_loss.detach().item()),
        finite_difference_epsilon=float(epsilon),
        delta_norm=delta_norm,
        train_loss_plus=float(loss_plus.detach().item()),
        train_loss_minus=float(loss_minus.detach().item()),
    )
