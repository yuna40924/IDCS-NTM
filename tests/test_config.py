import pytest

from idcs_ntm.engine import ExperimentConfig
from idcs_ntm.utils import set_classifier_learning_rate


def test_default_cifar100_configuration_is_valid():
    config = ExperimentConfig(
        dataset="cifar100",
        noise_type="asymmetric_i",
        noise_rate=0.4,
        method="idcs_ntm",
    )
    config.validate()


def test_dataset_rejects_the_other_datasets_noise_family():
    config = ExperimentConfig(
        dataset="cifar10",
        noise_type="asymmetric_i",
        noise_rate=0.4,
        method="ce",
    )
    with pytest.raises(ValueError, match="invalid for cifar10"):
        config.validate()


def test_learning_rate_matches_the_two_paper_milestones():
    import torch

    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    assert set_classifier_learning_rate(optimizer, epoch=79, initial=0.1) == pytest.approx(0.1)
    assert set_classifier_learning_rate(optimizer, epoch=80, initial=0.1) == pytest.approx(0.01)
    assert set_classifier_learning_rate(optimizer, epoch=100, initial=0.1) == pytest.approx(0.001)
