from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    allow_tf32: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = allow_tf32


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def set_classifier_learning_rate(
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    initial: float,
) -> float:
    factor = 1.0
    if epoch >= 80:
        factor *= 0.1
    if epoch >= 100:
        factor *= 0.1
    learning_rate = initial * factor
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def append_jsonl(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def default_num_workers() -> int:
    if os.name == "nt":
        return 2
    return min(4, os.cpu_count() or 1)
