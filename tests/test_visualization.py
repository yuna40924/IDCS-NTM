import json
import pickle

import numpy as np
from PIL import Image

from idcs_ntm.visualization import (
    adjusted_rand_index,
    align_learned_to_ground_truth,
    build_comparison,
    superclass_ids,
)


def test_alignment_recovers_permuted_superclass_ids():
    ground_truth = [list(range(start, start + 5)) for start in range(0, 100, 5)]
    learned = list(reversed(ground_truth))
    aligned, learned_to_ground_truth, contingency = align_learned_to_ground_truth(
        learned, ground_truth
    )
    np.testing.assert_array_equal(aligned, superclass_ids(ground_truth))
    assert learned_to_ground_truth.tolist() == list(reversed(range(20)))
    assert contingency.sum() == 100
    assert adjusted_rand_index(superclass_ids(learned), superclass_ids(ground_truth)) == 1.0


def test_adjusted_rand_index_detects_a_mismatch():
    first = np.repeat(np.arange(20), 5)
    second = first.copy()
    second[[0, 5]] = second[[5, 0]]
    assert adjusted_rand_index(first, second) < 1.0


def test_build_comparison_writes_png_and_json(tmp_path):
    ground_truth = [list(range(start, start + 5)) for start in range(0, 100, 5)]
    cifar_folder = tmp_path / "data" / "cifar-100-python"
    cifar_folder.mkdir(parents=True)
    with (cifar_folder / "train").open("wb") as handle:
        pickle.dump(
            {
                "fine_labels": list(range(100)),
                "coarse_labels": np.repeat(np.arange(20), 5).tolist(),
            },
            handle,
        )
    with (cifar_folder / "meta").open("wb") as handle:
        pickle.dump(
            {
                "fine_label_names": [f"fine_{index}" for index in range(100)],
                "coarse_label_names": [f"coarse_{index}" for index in range(20)],
            },
            handle,
        )
    run_dir = tmp_path / "outputs" / "cifar100" / "asymmetric_i_0p40" / "seed_1" / "idcs_ntm"
    run_dir.mkdir(parents=True)
    with (run_dir / "superclasses.json").open("w", encoding="utf-8") as handle:
        json.dump(list(reversed(ground_truth)), handle)

    report = build_comparison(
        run_dir=run_dir,
        data_root=tmp_path / "data",
        output_path=None,
        title=None,
    )

    assert report["optimal_matched_class_accuracy"] == 1.0
    assert report["adjusted_rand_index"] == 1.0
    with Image.open(run_dir / "superclass_comparison.png") as image:
        assert image.size == (1610, 350)
    assert (run_dir / "superclass_comparison.json").is_file()
