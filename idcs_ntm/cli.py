#解析参数、构造 ExperimentConfig、调用 run_experiment()
from __future__ import annotations

import argparse
from pathlib import Path

from .engine import ExperimentConfig, run_experiment
from .utils import default_num_workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce IDCS-NTM Section 6.3.1 on CIFAR"
    )
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument(
        "--noise-type",
        choices=("symmetric", "asymmetric", "asymmetric_i", "asymmetric_ii"),
        required=True,
    )
    parser.add_argument("--noise-rate", type=float, required=True)
    parser.add_argument("--method", choices=("ce", "forward", "idcs_ntm"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/section_6_3_1")
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--ce-checkpoint", type=Path)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--allow-tf32", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--classifier-lr", type=float, default=0.1)
    parser.add_argument("--classifier-momentum", type=float, default=0.9)
    parser.add_argument("--classifier-weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--transition-lr", type=float, default=1e-4)
    parser.add_argument("--transition-weight-decay", type=float, default=1e-4)
    parser.add_argument("--meta-interval", type=int, default=10)
    parser.add_argument("--meta-per-class", type=int, default=10)
    parser.add_argument("--meta-mixup-alpha", type=float, default=1.0)
    parser.add_argument(
        "--meta-target", choices=("observed", "prediction", "clean"), default="observed"
    )
    parser.add_argument("--finite-difference-scale", type=float, default=0.01)
    parser.add_argument("--c0", type=int, default=5)
    parser.add_argument(
        "--superclass-mode", choices=("visual", "official"), default="visual"
    )
    parser.add_argument(
        "--symmetric-include-self", action=argparse.BooleanOptionalAction, default=True
    )
    outlier = parser.add_mutually_exclusive_group()
    outlier.add_argument(
        "--forward-filter-outliers", dest="forward_filter_outliers", action="store_true"
    )
    outlier.add_argument(
        "--no-forward-filter-outliers",
        dest="forward_filter_outliers",
        action="store_false",
    )
    parser.set_defaults(forward_filter_outliers=None)
    parser.add_argument("--evaluate-every", type=int, default=1)
    parser.add_argument(
        "--save-checkpoint", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    arguments = vars(build_parser().parse_args())
    config = ExperimentConfig(**arguments)
    summary = run_experiment(config)
    print(
        f"completed {summary['method']} {summary['dataset']} "
        f"{summary['noise_type']}={summary['noise_rate_nominal']:.2f}: "
        f"{summary['final_test_accuracy']:.2f}%"
    )


if __name__ == "__main__":
    main()
