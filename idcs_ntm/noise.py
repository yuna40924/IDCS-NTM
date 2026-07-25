from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


CIFAR10_ASYMMETRIC_TARGET = np.asarray(
    [0, 1, 0, 5, 7, 3, 6, 7, 8, 1], dtype=np.int64
)


@dataclass(frozen=True)
class NoiseResult:
    noisy_labels: np.ndarray
    transition: np.ndarray
    selected_mask: np.ndarray
    nominal_rate: float
    actual_rate: float


def _validate_rate(rate: float) -> None:
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"noise rate must be in [0, 1], got {rate}")


def validate_superclasses(
    superclasses: Sequence[Sequence[int]], num_classes: int
) -> list[list[int]]:
    groups = [sorted(int(label) for label in group) for group in superclasses]
    flattened = [label for group in groups for label in group]
    if sorted(flattened) != list(range(num_classes)):
        raise ValueError(
            "superclasses must be a disjoint partition of "
            f"0..{num_classes - 1}; received {groups}"
        )
    if any(not group for group in groups):
        raise ValueError("superclasses cannot contain an empty group")
    return groups


def symmetric_transition(
    num_classes: int, rate: float, *, include_self: bool = True
) -> np.ndarray:
    """Return the nominal symmetric transition used by Figure 6-3.

    With ``include_self=True``, a fraction ``rate`` is selected and assigned a
    uniformly sampled label from all C labels.  Therefore the diagonal is
    ``1-rate+rate/C`` rather than ``1-rate``.  This is the convention visible
    in Figure 6-3 and in the Chapter 4 reference implementation.
    """

    _validate_rate(rate)
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if include_self:
        matrix = np.full(
            (num_classes, num_classes), rate / num_classes, dtype=np.float64
        )
        np.fill_diagonal(matrix, 1.0 - rate + rate / num_classes)
    else:
        matrix = np.full(
            (num_classes, num_classes), rate / (num_classes - 1), dtype=np.float64
        )
        np.fill_diagonal(matrix, 1.0 - rate)
    return matrix


def cifar10_asymmetric_transition(rate: float) -> np.ndarray:
    """Patrini-style CIFAR-10 asymmetric transition.

    truck -> automobile, bird -> airplane, deer -> horse, cat <-> dog.
    Classes not listed in those mappings remain clean.
    """

    _validate_rate(rate)
    matrix = np.zeros((10, 10), dtype=np.float64)
    for source, target in enumerate(CIFAR10_ASYMMETRIC_TARGET):
        if source == target:
            matrix[source, source] = 1.0
        else:
            matrix[source, source] = 1.0 - rate
            matrix[source, target] = rate
    return matrix


def superclass_symmetric_transition(
    num_classes: int,
    rate: float,
    superclasses: Sequence[Sequence[int]],
    *,
    include_self: bool = True,
) -> np.ndarray:
    """CIFAR-100 asymmetric-I: symmetric replacement inside each superclass."""

    _validate_rate(rate)
    groups = validate_superclasses(superclasses, num_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for group in groups:
        block = symmetric_transition(len(group), rate, include_self=include_self)
        matrix[np.ix_(group, group)] = block
    return matrix


def superclass_cyclic_transition(
    num_classes: int,
    rate: float,
    superclasses: Sequence[Sequence[int]],
) -> np.ndarray:
    """CIFAR-100 asymmetric-II: cyclic pair flipping within each superclass."""

    _validate_rate(rate)
    groups = validate_superclasses(superclasses, num_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    for group in groups:
        for position, source in enumerate(group):
            target = group[(position + 1) % len(group)]
            matrix[source, source] = 1.0 - rate
            matrix[source, target] += rate
    return matrix


def _select_exact_fraction(size: int, rate: float, rng: np.random.Generator) -> np.ndarray:
    count = int(rate * size)
    mask = np.zeros(size, dtype=bool)
    if count:
        mask[rng.permutation(size)[:count]] = True
    return mask


def _label_to_group(
    groups: Sequence[Sequence[int]], num_classes: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    lookup = np.full(num_classes, -1, dtype=np.int64)
    arrays: list[np.ndarray] = []
    for group_id, group in enumerate(groups):
        array = np.asarray(group, dtype=np.int64)
        arrays.append(array)
        lookup[array] = group_id
    return lookup, arrays


def inject_feature_independent_noise(
    clean_labels: Iterable[int] | np.ndarray,
    *,
    num_classes: int,
    noise_type: str,
    rate: float,
    seed: int,
    superclasses: Sequence[Sequence[int]] | None = None,
    symmetric_include_self: bool = True,
) -> NoiseResult:
    """Inject one of the feature-independent noises from Section 6.3.1.

    The implementation selects exactly ``floor(rate * N)`` training examples,
    matching the thesis wording and the Chapter 4 implementation.  The
    returned transition matrix is the nominal population matrix; an individual
    finite sample naturally has a slightly different empirical matrix.
    """

    _validate_rate(rate)
    clean = np.asarray(list(clean_labels), dtype=np.int64)
    if clean.ndim != 1:
        raise ValueError("clean_labels must be one-dimensional")
    if clean.size and (clean.min() < 0 or clean.max() >= num_classes):
        raise ValueError("clean_labels contain a class outside the configured range")

    kind = noise_type.lower().replace("-", "_")
    aliases = {
        "sym": "symmetric",
        "asym": "asymmetric",
        "asymmetric1": "asymmetric_i",
        "asymmetric_1": "asymmetric_i",
        "asymmetric2": "asymmetric_ii",
        "asymmetric_2": "asymmetric_ii",
    }
    kind = aliases.get(kind, kind)
    rng = np.random.default_rng(seed)
    selected = _select_exact_fraction(clean.size, rate, rng)
    noisy = clean.copy()

    if kind == "symmetric":
        transition = symmetric_transition(
            num_classes, rate, include_self=symmetric_include_self
        )
        if symmetric_include_self:
            noisy[selected] = rng.integers(0, num_classes, selected.sum())
        else:
            offsets = rng.integers(1, num_classes, selected.sum())
            noisy[selected] = (clean[selected] + offsets) % num_classes
    elif kind == "asymmetric":
        if num_classes != 10:
            raise ValueError("noise_type='asymmetric' is the CIFAR-10 mapping")
        transition = cifar10_asymmetric_transition(rate)
        noisy[selected] = CIFAR10_ASYMMETRIC_TARGET[clean[selected]]
    elif kind in {"asymmetric_i", "asymmetric_ii"}:
        if superclasses is None:
            raise ValueError(f"{kind} requires CIFAR-100 superclasses")
        groups = validate_superclasses(superclasses, num_classes)
        label_group, group_arrays = _label_to_group(groups, num_classes)
        if kind == "asymmetric_i":
            transition = superclass_symmetric_transition(
                num_classes,
                rate,
                groups,
                include_self=symmetric_include_self,
            )
            selected_indices = np.flatnonzero(selected)
            for index in selected_indices:
                candidates = group_arrays[label_group[clean[index]]]
                if symmetric_include_self:
                    noisy[index] = rng.choice(candidates)
                else:
                    candidates = candidates[candidates != clean[index]]
                    noisy[index] = rng.choice(candidates)
        else:
            transition = superclass_cyclic_transition(num_classes, rate, groups)
            next_label = np.empty(num_classes, dtype=np.int64)
            for group in group_arrays:
                next_label[group] = np.roll(group, -1)
            noisy[selected] = next_label[clean[selected]]
    else:
        raise ValueError(
            "noise_type must be symmetric, asymmetric, asymmetric_i, or "
            f"asymmetric_ii; got {noise_type!r}"
        )

    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
        raise RuntimeError("constructed transition is not row stochastic")
    actual_rate = float(np.mean(noisy != clean)) if clean.size else 0.0
    return NoiseResult(
        noisy_labels=noisy,
        transition=transition,
        selected_mask=selected,
        nominal_rate=float(rate),
        actual_rate=actual_rate,
    )
