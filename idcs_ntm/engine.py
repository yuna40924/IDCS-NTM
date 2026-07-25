#核心训练调度器；创建数据加载器并分派 CE、Forward 或 IDCS-NTM
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CifarBundle, load_cifar_bundle
from .forward import ModelOutputs, collect_model_outputs, estimate_anchor_transition
from .meta import (
    MetaSubset,
    cycle_loader,
    finite_difference_meta_step,
    frozen_parameters,
    select_balanced_low_loss_indices,
)
from .models import (
    IDCSTransition,
    cifar_resnet34,
    forward_corrected_nll,
    mask_transition,
)
from .models.transition import superclass_mask
from .superclasses import learn_visual_superclasses
from .utils import (
    append_jsonl,
    default_num_workers,
    json_ready,
    resolve_device,
    seed_everything,
    seed_worker,
    set_classifier_learning_rate,
    write_json,
)


@dataclass
class ExperimentConfig:
    dataset: Literal["cifar10", "cifar100"]
    noise_type: str
    noise_rate: float
    method: Literal["ce", "forward", "idcs_ntm"]
    seed: int = 1
    data_root: Path = Path("data")
    output_root: Path = Path("outputs/section_6_3_1")
    run_dir: Path | None = None
    ce_checkpoint: Path | None = None
    download: bool = True
    device: str = "auto"
    deterministic: bool = True
    allow_tf32: bool = False
    num_workers: int = default_num_workers()
    batch_size: int = 128
    eval_batch_size: int = 256
    epochs: int = 120
    classifier_lr: float = 0.1
    classifier_momentum: float = 0.9
    classifier_weight_decay: float = 1e-3
    hidden_dim: int = 100
    transition_lr: float = 1e-4
    transition_weight_decay: float = 1e-4
    meta_interval: int = 10
    meta_per_class: int = 10
    meta_mixup_alpha: float = 1.0
    meta_target: Literal["observed", "prediction", "clean"] = "observed"
    finite_difference_scale: float = 0.01
    c0: int = 5
    superclass_mode: Literal["visual", "official"] = "visual"
    symmetric_include_self: bool = True
    forward_filter_outliers: bool | None = None
    evaluate_every: int = 1
    save_checkpoint: bool = True
    overwrite: bool = False

    @property
    def num_classes(self) -> int:
        return 10 if self.dataset == "cifar10" else 100

    def resolved_run_dir(self) -> Path:
        if self.run_dir is not None:
            return Path(self.run_dir)
        rate_name = f"{self.noise_rate:.2f}".replace(".", "p")
        return (
            Path(self.output_root)
            / self.dataset
            / f"{self.noise_type}_{rate_name}"
            / f"seed_{self.seed}"
            / self.method
        )

    def validate(self) -> None:
        allowed_noise = {
            "cifar10": {"symmetric", "asymmetric"},
            "cifar100": {"symmetric", "asymmetric_i", "asymmetric_ii"},
        }
        if self.noise_type not in allowed_noise[self.dataset]:
            choices = ", ".join(sorted(allowed_noise[self.dataset]))
            raise ValueError(
                f"noise_type={self.noise_type!r} is invalid for {self.dataset}; "
                f"choose one of: {choices}"
            )
        if not 0.0 <= self.noise_rate <= 1.0:
            raise ValueError("noise_rate must be in [0, 1]")
        positive_fields = {
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "epochs": self.epochs,
            "meta_interval": self.meta_interval,
            "meta_per_class": self.meta_per_class,
            "c0": self.c0,
            "evaluate_every": self.evaluate_every,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.meta_mixup_alpha < 0:
            raise ValueError("meta_mixup_alpha cannot be negative")
        if self.finite_difference_scale <= 0:
            raise ValueError("finite_difference_scale must be positive")


def _loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    config: ExperimentConfig,
    seed_offset: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=resolve_device(config.device).type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for images, _observed, clean, _indices in loader:
        images = images.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        prediction = model(images).argmax(dim=1)
        correct += int((prediction == clean).sum().item())
        total += clean.numel()
    model.train(was_training)
    return 100.0 * correct / max(total, 1)


def _checkpoint_payload(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    epoch: int,
    test_accuracy: float,
) -> dict:
    return {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "test_accuracy": test_accuracy,
        "config": json_ready(asdict(config)),
    }


def _load_model_checkpoint(
    path: Path, *, num_classes: int, device: torch.device
) -> nn.Module:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload["model_state"] if "model_state" in payload else payload
    model = cifar_resnet34(num_classes)
    model.load_state_dict(state)
    return model.to(device)


def _train_fixed_transition(
    *,
    model: nn.Module,
    config: ExperimentConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
    transition: torch.Tensor | None,
) -> tuple[nn.Module, float]:
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.classifier_lr,
        momentum=config.classifier_momentum,
        weight_decay=config.classifier_weight_decay,
    )
    metrics_path = run_dir / "metrics.jsonl"
    final_accuracy = float("nan")
    for epoch in range(config.epochs):
        learning_rate = set_classifier_learning_rate(
            optimizer, epoch=epoch, initial=config.classifier_lr
        )
        model.train()
        loss_sum = 0.0
        example_count = 0
        for images, observed, _clean, _indices in train_loader:
            images = images.to(device, non_blocking=True)
            observed = observed.to(device, non_blocking=True)
            logits = model(images)
            if transition is None:
                loss = F.cross_entropy(logits, observed)
            else:
                loss = forward_corrected_nll(logits, observed, transition)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = images.shape[0]
            loss_sum += float(loss.detach().item()) * batch_size
            example_count += batch_size
        should_evaluate = (
            (epoch + 1) % config.evaluate_every == 0 or epoch + 1 == config.epochs
        )
        if should_evaluate:
            final_accuracy = evaluate(model, test_loader, device)
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train_loss": loss_sum / max(example_count, 1),
                "test_accuracy": final_accuracy if should_evaluate else None,
            },
        )
        print(
            f"[{config.method}] epoch {epoch + 1:03d}/{config.epochs} "
            f"loss={loss_sum / max(example_count, 1):.4f} "
            + (f"test={final_accuracy:.2f}" if should_evaluate else "")
        )
    if config.save_checkpoint:
        torch.save(
            _checkpoint_payload(
                model,
                config,
                epoch=config.epochs,
                test_accuracy=final_accuracy,
            ),
            run_dir / "checkpoint_last.pt",
        )
    return model, final_accuracy


def _train_anchor_if_needed(
    *,
    config: ExperimentConfig,
    bundle: CifarBundle,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
) -> nn.Module:
    if config.ce_checkpoint is not None:
        path = Path(config.ce_checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"CE checkpoint does not exist: {path}")
        return _load_model_checkpoint(path, num_classes=config.num_classes, device=device)

    anchor_dir = run_dir / "anchor_ce"
    anchor_path = anchor_dir / "checkpoint_last.pt"
    if anchor_path.exists():
        return _load_model_checkpoint(
            anchor_path, num_classes=config.num_classes, device=device
        )
    anchor_dir.mkdir(parents=True, exist_ok=True)
    anchor_config = ExperimentConfig(**{**asdict(config), "method": "ce", "run_dir": anchor_dir})
    seed_everything(
        config.seed,
        deterministic=config.deterministic,
        allow_tf32=config.allow_tf32,
    )
    anchor_model = cifar_resnet34(config.num_classes).to(device)
    anchor_model, _ = _train_fixed_transition(
        model=anchor_model,
        config=anchor_config,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        run_dir=anchor_dir,
        transition=None,
    )
    return anchor_model


def _meta_targets(
    config: ExperimentConfig,
    bundle: CifarBundle,
    outputs: ModelOutputs,
    indices: np.ndarray,
) -> np.ndarray:
    if config.meta_target == "observed":
        return bundle.noisy_labels[indices]
    if config.meta_target == "prediction":
        return outputs.probabilities[indices].argmax(axis=1)
    if config.meta_target == "clean":
        return bundle.clean_labels[indices]
    raise ValueError(f"unsupported meta_target {config.meta_target}")


def _build_meta_loader(
    *,
    config: ExperimentConfig,
    bundle: CifarBundle,
    outputs: ModelOutputs,
    epoch: int,
) -> tuple[DataLoader, np.ndarray, float]:
    indices = select_balanced_low_loss_indices(
        outputs.losses,
        bundle.noisy_labels,
        num_classes=config.num_classes,
        per_class=config.meta_per_class,
    )
    targets = _meta_targets(config, bundle, outputs, indices)
    precision = float(np.mean(targets == bundle.clean_labels[indices]))
    dataset = MetaSubset(bundle.train, indices, targets)
    loader = _loader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        config=config,
        seed_offset=20_000 + epoch,
    )
    return loader, indices, precision


@torch.no_grad()
def _average_idcs_transition(
    classifier: nn.Module,
    transition_model: IDCSTransition,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    classifier.eval()
    transition_model.eval()
    total = torch.zeros(
        transition_model.num_classes,
        transition_model.num_classes,
        device=device,
    )
    count = 0
    for images, _observed, _clean, _indices in loader:
        images = images.to(device, non_blocking=True)
        _logits, features = classifier(images, return_features=True)
        matrices = transition_model(features)
        total += matrices.sum(dim=0)
        count += images.shape[0]
    return (total / max(count, 1)).cpu().numpy()


def _train_idcs(
    *,
    model: nn.Module,
    transition_model: IDCSTransition,
    config: ExperimentConfig,
    bundle: CifarBundle,
    initial_outputs: ModelOutputs,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
) -> tuple[nn.Module, float]:
    classifier_optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.classifier_lr,
        momentum=config.classifier_momentum,
        weight_decay=config.classifier_weight_decay,
    )
    transition_optimizer = torch.optim.Adam(
        transition_model.parameters(),
        lr=config.transition_lr,
        weight_decay=config.transition_weight_decay,
    )
    metrics_path = run_dir / "metrics.jsonl"
    current_outputs = initial_outputs
    global_step = 0
    final_accuracy = float("nan")

    for epoch in range(config.epochs):
        learning_rate = set_classifier_learning_rate(
            classifier_optimizer, epoch=epoch, initial=config.classifier_lr
        )
        meta_loader, meta_indices, meta_precision = _build_meta_loader(
            config=config,
            bundle=bundle,
            outputs=current_outputs,
            epoch=epoch,
        )
        meta_iterator = cycle_loader(meta_loader)
        model.train()
        transition_model.train()
        loss_sum = 0.0
        example_count = 0
        meta_loss_sum = 0.0
        meta_updates = 0
        epsilon_sum = 0.0

        for images, observed, _clean, _indices in train_loader:
            images = images.to(device, non_blocking=True)
            observed = observed.to(device, non_blocking=True)
            if global_step % config.meta_interval == 0:
                meta_images, meta_targets, _meta_indices = next(meta_iterator)
                meta_images = meta_images.to(device, non_blocking=True)
                meta_targets = meta_targets.to(device, non_blocking=True)
                diagnostics = finite_difference_meta_step(
                    classifier=model,
                    transition_model=transition_model,
                    transition_optimizer=transition_optimizer,
                    train_images=images,
                    train_targets=observed,
                    meta_images=meta_images,
                    meta_targets=meta_targets,
                    inner_learning_rate=learning_rate,
                    finite_difference_scale=config.finite_difference_scale,
                    mixup_alpha=config.meta_mixup_alpha,
                )
                meta_loss_sum += diagnostics.meta_loss
                epsilon_sum += diagnostics.finite_difference_epsilon
                meta_updates += 1

            classifier_optimizer.zero_grad(set_to_none=True)
            with frozen_parameters(transition_model):
                logits, features = model(images, return_features=True)
                matrices = transition_model(features)
                loss = forward_corrected_nll(logits, observed, matrices)
                loss.backward()
            classifier_optimizer.step()
            batch_size = images.shape[0]
            loss_sum += float(loss.detach().item()) * batch_size
            example_count += batch_size
            global_step += 1

        should_evaluate = (
            (epoch + 1) % config.evaluate_every == 0 or epoch + 1 == config.epochs
        )
        if should_evaluate:
            final_accuracy = evaluate(model, test_loader, device)
        current_outputs = collect_model_outputs(
            model, eval_loader, device=device, collect_features=False
        )
        append_jsonl(
            metrics_path,
            {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train_loss": loss_sum / max(example_count, 1),
                "test_accuracy": final_accuracy if should_evaluate else None,
                "meta_loss": meta_loss_sum / max(meta_updates, 1),
                "meta_updates": meta_updates,
                "finite_difference_epsilon": epsilon_sum / max(meta_updates, 1),
                "meta_precision_diagnostic": meta_precision,
                "meta_size": len(meta_indices),
            },
        )
        print(
            f"[idcs_ntm] epoch {epoch + 1:03d}/{config.epochs} "
            f"loss={loss_sum / max(example_count, 1):.4f} "
            f"meta_precision={meta_precision:.3f} "
            + (f"test={final_accuracy:.2f}" if should_evaluate else "")
        )

    average_transition = _average_idcs_transition(
        model, transition_model, eval_loader, device
    )
    np.save(run_dir / "learned_transition_mean.npy", average_transition)
    np.save(run_dir / "learned_rho.npy", transition_model.prior().detach().cpu().numpy())
    if config.save_checkpoint:
        payload = _checkpoint_payload(
            model,
            config,
            epoch=config.epochs,
            test_accuracy=final_accuracy,
        )
        payload["transition_state"] = transition_model.state_dict()
        torch.save(payload, run_dir / "checkpoint_last.pt")
    return model, final_accuracy


def run_experiment(config: ExperimentConfig) -> dict:
    start_time = time.time()
    config.validate()
    run_dir = config.resolved_run_dir()
    existing_artifacts = [run_dir / "summary.json", run_dir / "metrics.jsonl"]
    if any(path.exists() for path in existing_artifacts) and not config.overwrite:
        raise FileExistsError(
            f"run directory already contains results: {run_dir}. "
            "Use --overwrite to replace this exact run."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if config.overwrite and metrics_path.exists():
        metrics_path.unlink()
    write_json(run_dir / "config.json", asdict(config))

    seed_everything(
        config.seed,
        deterministic=config.deterministic,
        allow_tf32=config.allow_tf32,
    )
    device = resolve_device(config.device)
    bundle = load_cifar_bundle(
        dataset=config.dataset,
        root=config.data_root,
        noise_type=config.noise_type,
        rate=config.noise_rate,
        seed=config.seed,
        download=config.download,
        symmetric_include_self=config.symmetric_include_self,
    )
    np.savez_compressed(
        run_dir / "noise_manifest.npz",
        clean_labels=bundle.clean_labels,
        noisy_labels=bundle.noisy_labels,
        selected_mask=bundle.selected_mask,
        true_transition=bundle.true_transition,
    )
    train_loader = _loader(
        bundle.train,
        batch_size=config.batch_size,
        shuffle=True,
        config=config,
        seed_offset=1_000,
    )
    eval_loader = _loader(
        bundle.eval_train,
        batch_size=config.eval_batch_size,
        shuffle=False,
        config=config,
        seed_offset=2_000,
    )
    test_loader = _loader(
        bundle.test,
        batch_size=config.eval_batch_size,
        shuffle=False,
        config=config,
        seed_offset=3_000,
    )

    if config.method == "ce":
        seed_everything(
            config.seed,
            deterministic=config.deterministic,
            allow_tf32=config.allow_tf32,
        )
        model = cifar_resnet34(config.num_classes).to(device)
        _model, final_accuracy = _train_fixed_transition(
            model=model,
            config=config,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            run_dir=run_dir,
            transition=None,
        )
        estimated_transition = None
        superclasses = None
    else:
        anchor_model = _train_anchor_if_needed(
            config=config,
            bundle=bundle,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            run_dir=run_dir,
        )
        filter_outliers = config.forward_filter_outliers
        if filter_outliers is None:
            filter_outliers = config.dataset == "cifar10"
        estimated_transition, anchors, anchor_outputs = estimate_anchor_transition(
            anchor_model,
            eval_loader,
            device=device,
            filter_outliers=filter_outliers,
        )
        np.save(run_dir / "forward_transition.npy", estimated_transition)
        np.save(run_dir / "forward_anchor_indices.npy", anchors)

        seed_everything(
            config.seed,
            deterministic=config.deterministic,
            allow_tf32=config.allow_tf32,
        )
        model = cifar_resnet34(config.num_classes).to(device)
        if config.method == "forward":
            fixed_transition = torch.as_tensor(
                estimated_transition, dtype=torch.float32, device=device
            )
            _model, final_accuracy = _train_fixed_transition(
                model=model,
                config=config,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                run_dir=run_dir,
                transition=fixed_transition,
            )
            superclasses = None
        elif config.method == "idcs_ntm":
            if config.dataset == "cifar10":
                superclasses = [list(range(10))]
            elif config.superclass_mode == "official":
                if bundle.official_superclasses is None:
                    raise RuntimeError("official CIFAR-100 superclasses are unavailable")
                superclasses = bundle.official_superclasses
            else:
                if anchor_outputs.features is None:
                    raise RuntimeError("visual features were not collected")
                superclasses = learn_visual_superclasses(
                    anchor_outputs.features,
                    bundle.noisy_labels,
                    num_classes=config.num_classes,
                    c0=config.c0,
                )
            write_json(run_dir / "superclasses.json", superclasses)
            mask = superclass_mask(config.num_classes, superclasses)
            initial_block_transition = mask_transition(estimated_transition, mask)
            np.save(
                run_dir / "forward_transition_blocked.npy",
                initial_block_transition.numpy(),
            )
            transition_model = IDCSTransition(
                feature_dim=model.feature_dim,
                hidden_dim=config.hidden_dim,
                num_classes=config.num_classes,
                superclasses=superclasses,
                initial_transition=initial_block_transition,
            ).to(device)
            _model, final_accuracy = _train_idcs(
                model=model,
                transition_model=transition_model,
                config=config,
                bundle=bundle,
                initial_outputs=anchor_outputs,
                train_loader=train_loader,
                eval_loader=eval_loader,
                test_loader=test_loader,
                device=device,
                run_dir=run_dir,
            )
        else:
            raise ValueError(f"unknown method {config.method}")

    summary = {
        "dataset": config.dataset,
        "noise_type": config.noise_type,
        "noise_rate_nominal": config.noise_rate,
        "noise_rate_actual": bundle.actual_noise_rate,
        "method": config.method,
        "seed": config.seed,
        "final_test_accuracy": final_accuracy,
        "elapsed_seconds": time.time() - start_time,
        "run_dir": run_dir,
        "forward_transition_available": estimated_transition is not None,
        "superclass_count": len(superclasses) if superclasses is not None else None,
        "superclass_sizes": (
            [len(group) for group in superclasses] if superclasses is not None else None
        ),
        "superclass_algorithm": (
            "capacity_constrained_divisive_two_means"
            if superclasses is not None and config.dataset == "cifar100"
            else None
        ),
    }
    write_json(run_dir / "summary.json", summary)
    return summary
