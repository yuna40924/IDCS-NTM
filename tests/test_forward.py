import numpy as np

from idcs_ntm.forward import estimate_anchor_transition_from_probabilities


def test_anchor_transition_takes_one_probability_row_per_class():
    probabilities = np.asarray(
        [[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]], dtype=np.float32
    )
    transition, anchors = estimate_anchor_transition_from_probabilities(
        probabilities, filter_outliers=False
    )
    np.testing.assert_array_equal(anchors, [0, 1])
    np.testing.assert_allclose(transition, probabilities[:2], atol=1e-6)
