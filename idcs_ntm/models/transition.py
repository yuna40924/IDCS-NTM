from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from idcs_ntm.noise import validate_superclasses


def superclass_mask(
    num_classes: int, superclasses: Sequence[Sequence[int]]
) -> Tensor:
    groups = validate_superclasses(superclasses, num_classes)
    mask = torch.zeros(num_classes, num_classes, dtype=torch.bool)
    for group in groups:
        labels = torch.as_tensor(group, dtype=torch.long)
        mask[labels[:, None], labels[None, :]] = True
    return mask


def mask_transition(
    transition: np.ndarray | Tensor,
    mask: Tensor,
    *,
    floor: float = 1e-6,
) -> Tensor:
    """Project a transition matrix onto superclass blocks and renormalize rows."""

    matrix = torch.as_tensor(transition, dtype=torch.float32).clone()
    if matrix.shape != mask.shape:
        raise ValueError(f"transition shape {matrix.shape} does not match mask {mask.shape}")
    matrix = torch.where(mask, matrix.clamp_min(floor), torch.zeros_like(matrix))
    row_sum = matrix.sum(dim=1, keepdim=True)
    if torch.any(row_sum <= 0):
        raise ValueError("at least one superclass-constrained transition row is empty")
    return matrix / row_sum


def forward_corrected_nll(
    clean_logits: Tensor,
    observed_targets: Tensor,
    transition: Tensor,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Forward loss for T[i,j] = P(observed=j | clean=i)."""

    clean_probability = F.softmax(clean_logits, dim=1)
    if transition.ndim == 2:
        noisy_probability = clean_probability @ transition
    elif transition.ndim == 3:
        noisy_probability = torch.bmm(clean_probability.unsqueeze(1), transition).squeeze(1)
    else:
        raise ValueError("transition must have shape [C,C] or [B,C,C]")
    selected = noisy_probability.gather(1, observed_targets[:, None]).squeeze(1)
    return -selected.clamp_min(eps).log().mean()


class IDCSTransition(nn.Module):
    """Equation (6-5): a visual-feature-conditioned block transition matrix.

    For allowed entries, logits are

        v_ij^T ReLU(U^T h(x)) + log(rho_ij),

    followed by a row-wise softmax.  ``log_rho`` is optimized instead of a raw
    probability so rho remains positive and every row remains normalized.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
        superclasses: Sequence[Sequence[int]],
        initial_transition: np.ndarray | Tensor,
        probability_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        mask = superclass_mask(num_classes, superclasses)
        prior = mask_transition(initial_transition, mask, floor=probability_floor)
        self.register_buffer("mask", mask)
        self.log_rho = nn.Parameter(torch.where(mask, prior.log(), torch.zeros_like(prior)))
        self.U = nn.Parameter(torch.empty(feature_dim, hidden_dim))
        self.V = nn.Parameter(torch.zeros(num_classes, num_classes, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound_u = 1.0 / math.sqrt(self.feature_dim)
        bound_v = 1.0 / math.sqrt(self.hidden_dim)
        nn.init.uniform_(self.U, -bound_u, bound_u)
        with torch.no_grad():
            self.V.uniform_(-bound_v, bound_v)
            self.V.mul_(self.mask.unsqueeze(-1))

    def prior(self) -> Tensor:
        logits = self.log_rho.masked_fill(~self.mask, -torch.inf)
        return F.softmax(logits, dim=1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"expected features [B,{self.feature_dim}], got {tuple(features.shape)}"
            )
        hidden = F.relu(features @ self.U)
        residual = torch.einsum("bh,ijh->bij", hidden, self.V)
        logits = residual + self.log_rho.unsqueeze(0)
        logits = logits.masked_fill(~self.mask.unsqueeze(0), -torch.inf)
        return F.softmax(logits, dim=2)

    @torch.no_grad()
    def enforce_mask_(self) -> None:
        self.V.mul_(self.mask.unsqueeze(-1))
        self.log_rho.masked_fill_(~self.mask, 0.0)
