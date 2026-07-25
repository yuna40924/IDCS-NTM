import numpy as np

from idcs_ntm.superclasses import divisive_visual_clustering, learn_visual_superclasses


def test_c0_five_produces_twenty_equal_cifar100_groups():
    rng = np.random.default_rng(3)
    centers = rng.normal(size=(100, 8))
    groups = divisive_visual_clustering(centers, max_size=5)
    assert len(groups) == 20
    assert {len(group) for group in groups} == {5}
    assert sorted(label for group in groups for label in group) == list(range(100))


def test_learning_uses_sample_features_and_observed_labels():
    labels = np.repeat(np.arange(10), 3)
    features = np.eye(10)[labels] + 0.001
    groups = learn_visual_superclasses(
        features, labels, num_classes=10, c0=5
    )
    assert len(groups) == 2
    assert all(len(group) == 5 for group in groups)


def test_divisive_split_keeps_two_obvious_visual_pairs_together():
    centers = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    groups = divisive_visual_clustering(centers, max_size=2)
    assert groups == [[0, 1], [2, 3]]
