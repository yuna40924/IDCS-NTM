import numpy as np
import torch

from idcs_ntm.models.transition import IDCSTransition, forward_corrected_nll


def test_idcs_transition_is_row_stochastic_and_block_sparse():
    prior = np.full((4, 4), 0.25, dtype=np.float32)
    model = IDCSTransition(
        feature_dim=3,
        hidden_dim=5,
        num_classes=4,
        superclasses=[[0, 1], [2, 3]],
        initial_transition=prior,
    )
    features = torch.randn(6, 3, requires_grad=True)
    transition = model(features)
    torch.testing.assert_close(transition.sum(dim=2), torch.ones(6, 4))
    assert torch.count_nonzero(transition[:, :2, 2:]) == 0
    assert torch.count_nonzero(transition[:, 2:, :2]) == 0
    transition[:, 0, 1].sum().backward()
    assert features.grad is not None


def test_forward_loss_uses_clean_rows_and_observed_columns():
    logits = torch.tensor([[12.0, -12.0]])
    target = torch.tensor([1])
    transition = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss = forward_corrected_nll(logits, target, transition)
    assert loss.item() < 1e-5
