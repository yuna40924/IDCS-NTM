#读取 YAML，枚举数据集、噪声率、种子和方法，通过子进程反复调用 cli.py
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml

from .report import aggregate_summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Section 6.3.1 sweep")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/section_6_3_1.yaml")
    )
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"))
    parser.add_argument("--noise-type")
    parser.add_argument(
        "--methods", nargs="+", choices=("ce", "forward", "idcs_ntm")
    )
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--download", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _common_arguments(defaults: dict) -> list[str]:
    mapping = {
        "data_root": "--data-root",
        "output_root": "--output-root",
        "device": "--device",
        "num_workers": "--num-workers",
        "batch_size": "--batch-size",
        "eval_batch_size": "--eval-batch-size",
        "epochs": "--epochs",
        "classifier_lr": "--classifier-lr",
        "classifier_momentum": "--classifier-momentum",
        "classifier_weight_decay": "--classifier-weight-decay",
        "hidden_dim": "--hidden-dim",
        "transition_lr": "--transition-lr",
        "transition_weight_decay": "--transition-weight-decay",
        "meta_interval": "--meta-interval",
        "meta_per_class": "--meta-per-class",
        "meta_mixup_alpha": "--meta-mixup-alpha",
        "meta_target": "--meta-target",
        "finite_difference_scale": "--finite-difference-scale",
        "c0": "--c0",
        "superclass_mode": "--superclass-mode",
        "evaluate_every": "--evaluate-every",
    }
    arguments: list[str] = []
    for key, flag in mapping.items():
        if key in defaults:
            arguments.extend((flag, str(defaults[key])))
    if defaults.get("download", True):
        arguments.append("--download")
    else:
        arguments.append("--no-download")
    if defaults.get("symmetric_include_self", True):
        arguments.append("--symmetric-include-self")
    else:
        arguments.append("--no-symmetric-include-self")
    if defaults.get("deterministic", True):
        arguments.append("--deterministic")
    else:
        arguments.append("--no-deterministic")
    if defaults.get("allow_tf32", False):
        arguments.append("--allow-tf32")
    else:
        arguments.append("--no-allow-tf32")
    if defaults.get("save_checkpoint", True):
        arguments.append("--save-checkpoint")
    else:
        arguments.append("--no-save-checkpoint")
    filter_outliers = defaults.get("forward_filter_outliers")
    if filter_outliers is True:
        arguments.append("--forward-filter-outliers")
    elif filter_outliers is False:
        arguments.append("--no-forward-filter-outliers")
    return arguments


def main() -> None:
    options = build_parser().parse_args()
    with options.config.open("r", encoding="utf-8") as handle:
        specification = yaml.safe_load(handle)
    defaults = dict(specification["defaults"])
    for key in ("data_root", "output_root", "device", "num_workers", "download"):
        value = getattr(options, key)
        if value is not None:
            defaults[key] = value
    seeds = options.seeds or specification["seeds"]
    methods = options.methods or specification["methods"]
    output_root = Path(defaults["output_root"])
    common = _common_arguments(defaults)

    for experiment in specification["experiments"]:
        dataset = experiment["dataset"]
        noise_type = experiment["noise_type"]
        if options.dataset and dataset != options.dataset:
            continue
        if options.noise_type and noise_type != options.noise_type:
            continue
        for rate in experiment["rates"]:
            rate_name = f"{float(rate):.2f}".replace(".", "p")
            for seed in seeds:
                ce_run = (
                    output_root
                    / dataset
                    / f"{noise_type}_{rate_name}"
                    / f"seed_{seed}"
                    / "ce"
                )
                ordered_methods = sorted(methods, key=lambda method: method != "ce")
                for method in ordered_methods:
                    run_dir = (
                        output_root
                        / dataset
                        / f"{noise_type}_{rate_name}"
                        / f"seed_{seed}"
                        / method
                    )
                    summary_path = run_dir / "summary.json"
                    if summary_path.exists() and not options.overwrite:
                        print(f"skip completed run: {run_dir}")
                        continue
                    restart_incomplete = (
                        (run_dir / "metrics.jsonl").exists()
                        and not summary_path.exists()
                        and not options.overwrite
                    )
                    command = [
                        sys.executable,
                        "-m",
                        "idcs_ntm.cli",
                        "--dataset",
                        dataset,
                        "--noise-type",
                        noise_type,
                        "--noise-rate",
                        str(rate),
                        "--method",
                        method,
                        "--seed",
                        str(seed),
                        *common,
                    ]
                    ce_checkpoint = ce_run / "checkpoint_last.pt"
                    if method != "ce" and ce_checkpoint.exists():
                        command.extend(("--ce-checkpoint", str(ce_checkpoint)))
                    if options.overwrite or restart_incomplete:
                        if restart_incomplete:
                            print(f"restart incomplete run: {run_dir}")
                        command.append("--overwrite")
                    print(" ".join(command))
                    if not options.dry_run:
                        subprocess.run(command, check=True)
    if not options.dry_run:
        aggregate_summaries(output_root, specification)


if __name__ == "__main__":
    main()
