from __future__ import annotations

from typing import Sequence

import numpy as np
from .noise import validate_superclasses


def compute_class_centers(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    *,
    normalize: bool = True,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("features must be [N,D] and labels must be [N]")
    centers = np.empty((num_classes, features.shape[1]), dtype=np.float64)
    for class_id in range(num_classes):
        members = features[labels == class_id]
        if not len(members):
            raise ValueError(f"class {class_id} has no samples for a visual center")
        centers[class_id] = members.mean(axis=0)
    if normalize:
        norms = np.linalg.norm(centers, axis=1, keepdims=True)
        centers = centers / np.maximum(norms, 1e-12)
    return centers


def _farthest_first(values: np.ndarray, count: int) -> np.ndarray:
    global_center = values.mean(axis=0, keepdims=True)
    first = int(np.argmax(((values - global_center) ** 2).sum(axis=1)))
    chosen = [first]
    min_distance = ((values - values[first]) ** 2).sum(axis=1)
    while len(chosen) < count:
        candidate = int(np.argmax(min_distance))
        chosen.append(candidate)
        distance = ((values - values[candidate]) ** 2).sum(axis=1)
        min_distance = np.minimum(min_distance, distance)
    return np.asarray(chosen, dtype=np.int64)


def _balanced_binary_split(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    left_size: int,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bisect one cluster while enforcing the requested child capacities."""

    subset = values[labels]
    if not 1 <= left_size < len(labels):
        raise ValueError("left_size must leave at least one class in each child")
    seeds = _farthest_first(subset, 2)
    centers = subset[seeds].copy()
    previous_left: np.ndarray | None = None

    for _ in range(max_iterations):
        squared_distance = ((subset[:, None, :] - centers[None, :, :]) ** 2).sum(
            axis=2
        )
        # Assign exactly ``left_size`` elements to child zero.  Sorting the
        # distance difference is the optimal two-cluster assignment for fixed
        # centroids and capacities.  The class id is a deterministic tie-break.
        preference = squared_distance[:, 0] - squared_distance[:, 1]
        order = np.lexsort((labels, preference))
        left_local = np.sort(order[:left_size])
        right_local = np.sort(order[left_size:])
        if previous_left is not None and np.array_equal(left_local, previous_left):
            break
        previous_left = left_local.copy()
        centers[0] = subset[left_local].mean(axis=0)
        centers[1] = subset[right_local].mean(axis=0)

    return labels[left_local], labels[right_local]


def _child_leaf_counts(size: int, max_size: int) -> tuple[int, int, int]:
    leaf_count = int(np.ceil(size / max_size))
    left_leaves = leaf_count // 2
    right_leaves = leaf_count - left_leaves
    desired = int(round(size * left_leaves / leaf_count))
    minimum = max(left_leaves, size - right_leaves * max_size)
    maximum = min(left_leaves * max_size, size - right_leaves)
    return left_leaves, right_leaves, int(np.clip(desired, minimum, maximum))


def divisive_visual_clustering(
    class_centers: np.ndarray,
    *,
    max_size: int,
    max_iterations: int = 100,
) -> list[list[int]]:
    """Paper Algorithm 6-1 with deterministic capacity-constrained bisection.

    Algorithm 6-1 specifies divisive hierarchical clustering but omits its split
    criterion.  Each oversized node is therefore bisected with two-means while
    constraining child sizes so they can end in leaves of capacity ``C0``.  For
    CIFAR-100 and ``C0=5`` this produces the 20 equal-size leaves stated in the
    experimental text, without using the official coarse labels.
    """

    values = np.asarray(class_centers, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("class_centers must have shape [C,D]")
    num_classes = values.shape[0]
    if not 1 <= max_size <= num_classes:
        raise ValueError("max_size must be in [1, num_classes]")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    pending = [np.arange(num_classes, dtype=np.int64)]
    leaves: list[np.ndarray] = []
    while pending:
        labels = pending.pop(0)
        if len(labels) <= max_size:
            leaves.append(labels)
            continue
        _left_leaves, _right_leaves, left_size = _child_leaf_counts(
            len(labels), max_size
        )
        left, right = _balanced_binary_split(
            values,
            labels,
            left_size=left_size,
            max_iterations=max_iterations,
        )
        pending.extend((left, right))

    groups = [sorted(group.tolist()) for group in leaves]
    groups.sort(key=lambda group: group[0])
    return validate_superclasses(groups, num_classes)


def balanced_visual_clustering(
    class_centers: np.ndarray,
    *,
    max_size: int,
    max_iterations: int = 100,
) -> list[list[int]]:
    """Backward-compatible name for the divisive Algorithm 6-1 implementation."""

    return divisive_visual_clustering(
        class_centers, max_size=max_size, max_iterations=max_iterations
    )


def learn_visual_superclasses(
    features: np.ndarray,
    observed_labels: np.ndarray,
    *,
    num_classes: int,
    c0: int,
) -> list[list[int]]:
    centers = compute_class_centers(features, observed_labels, num_classes)
    return divisive_visual_clustering(centers, max_size=c0)


def groups_to_ids(
    superclasses: Sequence[Sequence[int]], num_classes: int
) -> np.ndarray:
    groups = validate_superclasses(superclasses, num_classes)
    result = np.empty(num_classes, dtype=np.int64)
    for group_id, group in enumerate(groups):
        result[np.asarray(group, dtype=np.int64)] = group_id
    return result
