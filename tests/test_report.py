import json

import pytest

from idcs_ntm.report import aggregate_summaries


def test_report_adds_paper_comparison(tmp_path):
    run = tmp_path / "cifar10" / "symmetric_0p40" / "seed_1" / "ce"
    run.mkdir(parents=True)
    with (run / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "cifar10",
                "noise_type": "symmetric",
                "noise_rate_nominal": 0.4,
                "method": "ce",
                "final_test_accuracy": 78.0,
            },
            handle,
        )
    specification = {
        "experiments": [
            {"dataset": "cifar10", "noise_type": "symmetric", "rates": [0.4]}
        ],
        "paper_targets": {
            "cifar10": {"symmetric": {"ce": [[77.26, 0.54]]}}
        },
    }

    rows = aggregate_summaries(tmp_path, specification)

    assert rows[0]["paper_mean_accuracy"] == 77.26
    assert rows[0]["mean_minus_paper"] == pytest.approx(0.74)
    assert (tmp_path / "aggregate.csv").exists()
