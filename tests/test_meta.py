import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from idcs_ntm.meta import finite_difference_meta_step, select_balanced_low_loss_indices
from idcs_ntm.models.transition import IDCSTransition


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = nn.Linear(4, 3)
        self.fc = nn.Linear(3, 2)

    def forward(self, x, *, return_features=False):
        features = F.relu(self.feature(x))
        logits = self.fc(features)
        return (logits, features) if return_features else logits


def test_finite_difference_meta_step_updates_transition_parameters():
    torch.manual_seed(4)
    classifier = TinyClassifier()
    transition = IDCSTransition(
        feature_dim=3,
        hidden_dim=4,
        num_classes=2,
        superclasses=[[0, 1]],
        initial_transition=np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=np.float32),
    )
    optimizer = torch.optim.Adam(transition.parameters(), lr=1e-3)
    train_x = torch.randn(8, 4)
    train_y = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    meta_x = torch.randn(6, 4)
    meta_y = torch.tensor([0, 1, 0, 1, 0, 1])
    before = [parameter.detach().clone() for parameter in transition.parameters()]
    diagnostics = finite_difference_meta_step(
        classifier=classifier,
        transition_model=transition,
        transition_optimizer=optimizer,
        train_images=train_x,
        train_targets=train_y,
        meta_images=meta_x,
        meta_targets=meta_y,
        inner_learning_rate=0.1,
    )
    after = list(transition.parameters())
    assert diagnostics.delta_norm > 0
    assert any(not torch.equal(old, new.detach()) for old, new in zip(before, after))


def test_meta_selection_takes_lowest_losses_per_observed_class():
    losses = np.asarray([0.8, 0.1, 0.2, 0.7, 0.3, 0.4])
    observed = np.asarray([0, 0, 0, 1, 1, 1])
    selected = select_balanced_low_loss_indices(
        losses, observed, num_classes=2, per_class=2
    )
    np.testing.assert_array_equal(selected, [1, 2, 4, 5])
