import numpy as np

from idcs_ntm.noise import (
    cifar10_asymmetric_transition,
    inject_feature_independent_noise,
    superclass_cyclic_transition,
    superclass_symmetric_transition,
    symmetric_transition,
)


def test_figure_6_3_cifar10_symmetric_values():
    transition = symmetric_transition(10, 0.4, include_self=True)
    np.testing.assert_allclose(np.diag(transition), 0.64)
    np.testing.assert_allclose(transition[0, 1:], 0.04)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0)


def test_cifar10_asymmetric_mapping():
    transition = cifar10_asymmetric_transition(0.4)
    assert transition[9, 1] == 0.4  # truck -> automobile
    assert transition[2, 0] == 0.4  # bird -> airplane
    assert transition[4, 7] == 0.4  # deer -> horse
    assert transition[3, 5] == 0.4 and transition[5, 3] == 0.4
    assert transition[0, 0] == 1.0


def test_cifar100_block_noise_values():
    groups = [list(range(0, 5)), list(range(5, 10))]
    symmetric = superclass_symmetric_transition(10, 0.4, groups)
    np.testing.assert_allclose(symmetric[0, 0], 0.68)
    np.testing.assert_allclose(symmetric[0, 1], 0.08)
    assert symmetric[0, 5] == 0.0
    cyclic = superclass_cyclic_transition(10, 0.3, groups)
    np.testing.assert_allclose(cyclic[0, [0, 1]], [0.7, 0.3])
    np.testing.assert_allclose(cyclic[4, 0], 0.3)


def test_noise_sampling_is_exact_and_deterministic():
    labels = np.tile(np.arange(10), 100)
    first = inject_feature_independent_noise(
        labels,
        num_classes=10,
        noise_type="symmetric",
        rate=0.4,
        seed=7,
    )
    second = inject_feature_independent_noise(
        labels,
        num_classes=10,
        noise_type="symmetric",
        rate=0.4,
        seed=7,
    )
    assert first.selected_mask.sum() == 400
    np.testing.assert_array_equal(first.noisy_labels, second.noisy_labels)
