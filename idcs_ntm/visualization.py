from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
import pickle
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

from .noise import validate_superclasses


def _pickle_value(mapping: Mapping, key: str):
    if key in mapping:
        return mapping[key]
    encoded = key.encode("utf-8")
    if encoded in mapping:
        return mapping[encoded]
    raise KeyError(f"CIFAR pickle does not contain {key!r}")


def _decode_names(values: Sequence) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ]


def load_cifar100_structure(
    data_root: str | Path,
) -> tuple[list[list[int]], list[str], list[str]]:
    folder = Path(data_root) / "cifar-100-python"
    train_path = folder / "train"
    meta_path = folder / "meta"
    if not train_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            "expected CIFAR-100 files at "
            f"{train_path} and {meta_path}; pass their parent as --data-root"
        )
    with train_path.open("rb") as handle:
        train = pickle.load(handle, encoding="latin1")
    fine = np.asarray(_pickle_value(train, "fine_labels"), dtype=np.int64)
    coarse = np.asarray(_pickle_value(train, "coarse_labels"), dtype=np.int64)
    fine_to_coarse = np.full(100, -1, dtype=np.int64)
    for fine_id in range(100):
        values = np.unique(coarse[fine == fine_id])
        if values.size != 1:
            raise RuntimeError(f"fine class {fine_id} maps to {values.size} coarse classes")
        fine_to_coarse[fine_id] = int(values[0])
    groups = [
        np.flatnonzero(fine_to_coarse == coarse_id).tolist()
        for coarse_id in range(20)
    ]
    validate_superclasses(groups, 100)
    with meta_path.open("rb") as handle:
        metadata = pickle.load(handle, encoding="latin1")
    fine_names = _decode_names(_pickle_value(metadata, "fine_label_names"))
    coarse_names = _decode_names(_pickle_value(metadata, "coarse_label_names"))
    return groups, fine_names, coarse_names


def superclass_ids(groups: Sequence[Sequence[int]], num_classes: int = 100) -> np.ndarray:
    groups = validate_superclasses(groups, num_classes)
    result = np.full(num_classes, -1, dtype=np.int64)
    for group_id, group in enumerate(groups):
        result[np.asarray(group, dtype=np.int64)] = group_id
    return result


def align_learned_to_ground_truth(
    learned_groups: Sequence[Sequence[int]],
    ground_truth_groups: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    learned = validate_superclasses(learned_groups, 100)
    ground_truth = validate_superclasses(ground_truth_groups, 100)
    if len(learned) != len(ground_truth):
        raise ValueError(
            f"Figure 6-5 requires equal cluster counts; got {len(learned)} learned "
            f"and {len(ground_truth)} ground-truth groups"
        )
    learned_ids = superclass_ids(learned)
    ground_truth_ids = superclass_ids(ground_truth)
    contingency = np.zeros((len(learned), len(ground_truth)), dtype=np.int64)
    np.add.at(contingency, (learned_ids, ground_truth_ids), 1)
    learned_rows, ground_truth_columns = linear_sum_assignment(-contingency)
    learned_to_ground_truth = np.full(len(learned), -1, dtype=np.int64)
    learned_to_ground_truth[learned_rows] = ground_truth_columns
    aligned_ids = learned_to_ground_truth[learned_ids]
    return aligned_ids, learned_to_ground_truth, contingency


def adjusted_rand_index(first_ids: np.ndarray, second_ids: np.ndarray) -> float:
    first = np.asarray(first_ids, dtype=np.int64)
    second = np.asarray(second_ids, dtype=np.int64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("cluster id arrays must have the same one-dimensional shape")
    contingency = np.zeros((first.max() + 1, second.max() + 1), dtype=np.int64)
    np.add.at(contingency, (first, second), 1)

    def choose_two(values: np.ndarray) -> float:
        values = values.astype(np.float64)
        return float(np.sum(values * (values - 1.0) / 2.0))

    pairs = first.size * (first.size - 1.0) / 2.0
    if pairs == 0:
        return 1.0
    index = choose_two(contingency)
    first_pairs = choose_two(contingency.sum(axis=1))
    second_pairs = choose_two(contingency.sum(axis=0))
    expected = first_pairs * second_pairs / pairs
    maximum = 0.5 * (first_pairs + second_pairs)
    denominator = maximum - expected
    return 1.0 if denominator == 0 else float((index - expected) / denominator)


def _palette(size: int) -> list[tuple[int, int, int]]:
    colors = []
    for index in range(size):
        hue = (index * 0.618033988749895) % 1.0
        saturation = 0.62 if index % 2 == 0 else 0.42
        value = 0.82 if index % 3 else 0.68
        red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(red * 255), int(green * 255), int(blue * 255)))
    return colors


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_figure_6_5(
    *,
    ground_truth_groups: Sequence[Sequence[int]],
    aligned_learned_ids: np.ndarray,
    output_path: str | Path,
    title: str,
    matched_accuracy: float,
    ari: float,
) -> None:
    ground_truth_ids = superclass_ids(ground_truth_groups)
    class_order = np.asarray(
        [label for group in ground_truth_groups for label in sorted(group)],
        dtype=np.int64,
    )
    cell_width = 13
    row_height = 52
    left = 220
    top = 105
    plot_width = cell_width * 100
    canvas = Image.new("RGB", (left + plot_width + 90, 350), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(25)
    label_font = _font(19)
    small_font = _font(15)
    colors = _palette(20)
    draw.text((left, 35), title, fill="black", font=title_font)

    rows = (
        ("Ground-truth", ground_truth_ids[class_order]),
        ("Ours", aligned_learned_ids[class_order]),
    )
    for row_index, (label, identifiers) in enumerate(rows):
        y0 = top + row_index * (row_height + 18)
        draw.text((20, y0 + 13), label, fill="black", font=label_font)
        for position, group_id in enumerate(identifiers):
            x0 = left + position * cell_width
            draw.rectangle(
                (x0, y0, x0 + cell_width, y0 + row_height),
                fill=colors[int(group_id)],
            )
        draw.rectangle(
            (left, y0, left + plot_width, y0 + row_height),
            outline=(40, 40, 40),
            width=1,
        )
        for boundary in range(0, 101, 5):
            x = left + boundary * cell_width
            draw.line((x, y0, x, y0 + row_height), fill="white", width=2)

    group_y = top + 2 * (row_height + 18) + 3
    for group_id in range(20):
        center = left + (group_id * 5 + 2.5) * cell_width
        text = str(group_id + 1)
        box = draw.textbbox((0, 0), text, font=small_font)
        draw.text(
            (center - (box[2] - box[0]) / 2, group_y),
            text,
            fill=(60, 60, 60),
            font=small_font,
        )
    metrics = (
        f"Optimal matched class accuracy: {matched_accuracy:.1%}    "
        f"Adjusted Rand index: {ari:.4f}"
    )
    draw.text((left, 305), metrics, fill="black", font=label_font)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def build_comparison(
    *, run_dir: Path, data_root: Path, output_path: Path | None, title: str | None
) -> dict:
    superclass_path = run_dir / "superclasses.json"
    if not superclass_path.is_file():
        raise FileNotFoundError(f"IDCS superclass file does not exist: {superclass_path}")
    with superclass_path.open("r", encoding="utf-8") as handle:
        learned_groups = json.load(handle)
    flattened = [label for group in learned_groups for label in group]
    if len(flattened) != 100:
        raise ValueError(
            "Figure 6-5 is a CIFAR-100 comparison. This run contains "
            f"{len(flattened)} classes; CIFAR-10 intentionally uses one superclass."
        )
    ground_truth_groups, fine_names, coarse_names = load_cifar100_structure(data_root)
    aligned, learned_to_ground_truth, contingency = align_learned_to_ground_truth(
        learned_groups, ground_truth_groups
    )
    ground_truth_ids = superclass_ids(ground_truth_groups)
    matched_accuracy = float(np.mean(aligned == ground_truth_ids))
    ari = adjusted_rand_index(superclass_ids(learned_groups), ground_truth_ids)
    destination = output_path or run_dir / "superclass_comparison.png"
    default_title = f"Figure 6-5 style superclass comparison - {run_dir.parent.parent.name}"
    render_figure_6_5(
        ground_truth_groups=ground_truth_groups,
        aligned_learned_ids=aligned,
        output_path=destination,
        title=title or default_title,
        matched_accuracy=matched_accuracy,
        ari=ari,
    )
    learned_details = []
    for learned_id, group in enumerate(learned_groups):
        matched_id = int(learned_to_ground_truth[learned_id])
        learned_details.append(
            {
                "learned_group": learned_id,
                "matched_ground_truth_group": matched_id,
                "matched_coarse_name": coarse_names[matched_id],
                "overlap": int(contingency[learned_id, matched_id]),
                "class_ids": [int(label) for label in group],
                "class_names": [fine_names[int(label)] for label in group],
            }
        )
    report = {
        "run_dir": str(run_dir),
        "figure": str(destination),
        "optimal_matched_class_accuracy": matched_accuracy,
        "adjusted_rand_index": ari,
        "ground_truth_groups": [
            {
                "group": group_id,
                "coarse_name": coarse_names[group_id],
                "class_ids": group,
                "class_names": [fine_names[label] for label in group],
            }
            for group_id, group in enumerate(ground_truth_groups)
        ],
        "learned_groups": learned_details,
    }
    report_path = run_dir / "superclass_comparison.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"figure={destination}")
    print(f"report={report_path}")
    print(f"optimal_matched_class_accuracy={matched_accuracy:.6f}")
    print(f"adjusted_rand_index={ari:.6f}")
    for item in sorted(learned_details, key=lambda value: value["matched_ground_truth_group"]):
        print(
            f"GT {item['matched_ground_truth_group']:02d} "
            f"{item['matched_coarse_name']}: overlap={item['overlap']}/5; "
            + ", ".join(item["class_names"])
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a Figure 6-5 style CIFAR-100 superclass comparison"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    options = parser.parse_args()
    build_comparison(
        run_dir=options.run_dir,
        data_root=options.data_root,
        output_path=options.output,
        title=options.title,
    )


if __name__ == "__main__":
    main()

