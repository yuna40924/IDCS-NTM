#汇总所有 summary.json，计算三次实验均值/标准差，并与论文表格比较
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .utils import write_json


def collect_summaries(output_root: str | Path) -> list[dict]:
    summaries = []
    for path in Path(output_root).rglob("summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            item = json.load(handle)
        item["summary_path"] = str(path)
        summaries.append(item)
    return summaries


def _paper_lookup(specification: dict | None) -> dict[tuple, tuple[float, float]]:
    if specification is None:
        return {}
    rates_by_experiment = {
        (item["dataset"], item["noise_type"]): [float(rate) for rate in item["rates"]]
        for item in specification.get("experiments", [])
    }
    lookup: dict[tuple, tuple[float, float]] = {}
    for dataset, noise_families in specification.get("paper_targets", {}).items():
        for noise_type, methods in noise_families.items():
            rates = rates_by_experiment[(dataset, noise_type)]
            for method, values in methods.items():
                if len(values) != len(rates):
                    raise ValueError(
                        f"paper target length mismatch for {dataset}/{noise_type}/{method}"
                    )
                for rate, (mean, standard_deviation) in zip(rates, values):
                    lookup[(dataset, noise_type, rate, method)] = (
                        float(mean),
                        float(standard_deviation),
                    )
    return lookup


def aggregate_summaries(
    output_root: str | Path, specification: dict | None = None
) -> list[dict]:
    paper_lookup = _paper_lookup(specification)
    grouped: dict[tuple, list[float]] = {}
    for item in collect_summaries(output_root):
        key = (
            item["dataset"],
            item["noise_type"],
            float(item["noise_rate_nominal"]),
            item["method"],
        )
        grouped.setdefault(key, []).append(float(item["final_test_accuracy"]))
    rows = []
    for (dataset, noise_type, rate, method), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        mean_accuracy = float(array.mean())
        row = {
            "dataset": dataset,
            "noise_type": noise_type,
            "noise_rate": rate,
            "method": method,
            "runs": len(values),
            "mean_accuracy": mean_accuracy,
            "std_accuracy": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        }
        target = paper_lookup.get((dataset, noise_type, rate, method))
        if target is not None:
            row["paper_mean_accuracy"] = target[0]
            row["paper_std_accuracy"] = target[1]
            row["mean_minus_paper"] = mean_accuracy - target[0]
        rows.append(row)
    root = Path(output_root)
    write_json(root / "aggregate.json", rows)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "noise_type",
            "noise_rate",
            "method",
            "runs",
            "mean_accuracy",
            "std_accuracy",
        ]
        if any("paper_mean_accuracy" in row for row in rows):
            fieldnames.extend(
                ["paper_mean_accuracy", "paper_std_accuracy", "mean_minus_paper"]
            )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
