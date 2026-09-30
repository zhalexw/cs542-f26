# SYSTEM IMPORTS
from __future__ import annotations
import argparse as ap
import numpy as np
import os
import csv
from collections.abc import Callable, Sequence


# PYTHON PROJECT IMPORTS
from src.dt.clf import DecisionTreeClassifier, RandomForestClassifier
from src.dt.data.c45 import load_spam, load_volcanoes, load_voting
from src.dt.quality.gr import GainRatio
from src.dt.quality.ig import InformationGain




DATASETS = {
    "spam": load_spam,
    "volcanoes": load_volcanoes,
    "voting": load_voting,
}

QUALITY_FUNCTIONS = {
    "ig": InformationGain,
    "gr": GainRatio,
}




def stratified_kfold_indices(y_gt: np.ndarray,
                             num_folds: int,
                             seed: int) -> Sequence[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds = [list() for _ in range(num_folds)]

    for cls in np.unique(y_gt):
        class_idxs = np.flatnonzero(y_gt == cls)
        rng.shuffle(class_idxs)
        for fold_idx, fold_class_idxs in enumerate(np.array_split(class_idxs, num_folds)):
            folds[fold_idx].extend(fold_class_idxs.tolist())

    return [np.array(sorted(fold), dtype=int) for fold in folds]


def accuracy(y_hat: np.ndarray, y_gt: np.ndarray) -> float:
    return float(np.mean(y_hat == y_gt))


def cv_result(dataset_name: str,
              quality_name: str,
              max_depth: int,
              num_folds: int,
              seed: int) -> dict:
    header, X, y_gt = DATASETS[dataset_name]()
    folds = stratified_kfold_indices(y_gt, num_folds, seed)
    all_idxs = np.arange(y_gt.shape[0])
    fold_accuracies = list()

    for test_idxs in folds:
        train_idxs = np.setdiff1d(all_idxs, test_idxs, assume_unique=True)
        model = DecisionTreeClassifier(header, QUALITY_FUNCTIONS[quality_name]())
        model.fit(X[train_idxs], y_gt[train_idxs])
        fold_accuracies.append(accuracy(model.predict(X[test_idxs]), y_gt[test_idxs]))

    fold_accuracies = np.array(fold_accuracies)
    return {
        "dataset": dataset_name,
        "quality_function": quality_name,
        "max_depth": max_depth,
        "fold_accuracies": fold_accuracies.tolist(),
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_variance": float(np.var(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
    }


def full_training_result(dataset_name: str,
                         quality_name: str,
                         max_depth: int) -> dict:
    header, X, y_gt = DATASETS[dataset_name]()
    model = DecisionTreeClassifier(header, QUALITY_FUNCTIONS[quality_name]())
    model.fit(X, y_gt)
    return {
        "dataset": dataset_name,
        "quality_function": quality_name,
        "max_depth": max_depth,
        "training_accuracy": accuracy(model.predict(X), y_gt),
        "num_nodes": model.num_nodes,
    }


def write_csv(path: str, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_svg_plot(path: str,
                   title: str,
                   rows: Sequence[dict]) -> None:
    width, height = 720, 440
    left, right, top, bottom = 70, 30, 45, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    depths = [int(row["max_depth"]) for row in rows]
    means = [float(row["accuracy_mean"]) for row in rows]
    stds = [float(row["accuracy_std"]) for row in rows]
    min_depth, max_depth = min(depths), max(depths)

    def sx(depth: int) -> float:
        if min_depth == max_depth:
            return left + plot_w / 2
        return left + (depth - min_depth) / (max_depth - min_depth) * plot_w

    def sy(acc: float) -> float:
        return top + (1.0 - acc) * plot_h

    points = " ".join(f"{sx(d):.2f},{sy(m):.2f}" for d, m in zip(depths, means))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">max depth</text>',
        f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" font-family="Arial" font-size="13" transform="rotate(-90 18 {top + plot_h / 2})">accuracy</text>',
    ]

    for tick in range(0, 6):
        acc = tick / 5
        y = sy(acc)
        parts.append(f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#222"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="11">{acc:.1f}</text>')
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e8e8e8"/>')

    for depth in depths:
        x = sx(depth)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 5}" stroke="#222"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11">{depth}</text>')

    parts.append(f'<polyline points="{points}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>')
    for depth, mean, std in zip(depths, means, stds):
        x = sx(depth)
        y = sy(mean)
        y_low = sy(max(0.0, mean - std))
        y_high = sy(min(1.0, mean + std))
        parts.append(f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}" stroke="#555"/>')
        parts.append(f'<line x1="{x - 5:.2f}" y1="{y_low:.2f}" x2="{x + 5:.2f}" y2="{y_low:.2f}" stroke="#555"/>')
        parts.append(f'<line x1="{x - 5:.2f}" y1="{y_high:.2f}" x2="{x + 5:.2f}" y2="{y_high:.2f}" stroke="#555"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="#1f77b4"/>')

    parts.append("</svg>")
    with open(path, "w", newline="") as f:
        f.write("\n".join(parts))


def markdown_table(rows: Sequence[dict], fields: Sequence[str]) -> str:
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = list()
    for row in rows:
        values = list()
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                value = f"{value:.6f}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + body)


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-dir", default="task5_results")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS.keys(),
                        default=["voting", "spam", "volcanoes"])
    parser.add_argument("--depth-grid", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--quality-depths", nargs="+", type=int, default=[1, 3, 5])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cache = dict()

    def get_cv(dataset_name: str, quality_name: str, max_depth: int) -> dict:
        key = (dataset_name, quality_name, max_depth)
        if key not in cache:
            cache[key] = cv_result(dataset_name, quality_name, max_depth, args.folds, args.seed)
            print(
                f"{dataset_name} {quality_name} depth={max_depth}: "
                f"mean={cache[key]['accuracy_mean']:.4f}, "
                f"var={cache[key]['accuracy_variance']:.6f}",
                flush=True,
            )
        return cache[key]

    depth_one_rows = [get_cv(dataset, "ig", 1) for dataset in args.datasets]

    depth_grid_rows = [
        get_cv(dataset, "ig", depth)
        for dataset in args.datasets
        if dataset in {"spam", "volcanoes"}
        for depth in args.depth_grid
    ]

    quality_rows = [
        get_cv(dataset, quality_name, depth)
        for dataset in args.datasets
        for depth in args.quality_depths
        for quality_name in ["ig", "gr"]
    ]

    full_rows = [
        full_training_result(dataset, "ig", depth)
        for dataset in args.datasets
        for depth in [1, 2]
    ]

    cv_fields = [
        "dataset", "quality_function", "max_depth",
        "accuracy_mean", "accuracy_variance", "accuracy_std", "fold_accuracies",
    ]
    write_csv(os.path.join(args.output_dir, "depth_one_cv.csv"), depth_one_rows, cv_fields)
    write_csv(os.path.join(args.output_dir, "depth_grid_cv.csv"), depth_grid_rows, cv_fields)
    write_csv(os.path.join(args.output_dir, "quality_comparison_cv.csv"), quality_rows, cv_fields)
    write_csv(os.path.join(args.output_dir, "full_training_accuracy.csv"),
              full_rows,
              ["dataset", "quality_function", "max_depth", "training_accuracy", "num_nodes"])

    for dataset in ["spam", "volcanoes"]:
        rows = [row for row in depth_grid_rows if row["dataset"] == dataset]
        if rows:
            write_svg_plot(os.path.join(args.output_dir, f"{dataset}_depth_accuracy.svg"),
                           f"{dataset}: 5-fold CV accuracy by max depth", rows)

    summary_path = os.path.join(args.output_dir, "decision_tree_experiments.md")
    with open(summary_path, "w", newline="") as f:
        f.write("# Decision Tree Experiments\n\n")
        f.write("## Depth 1, Information Gain\n\n")
        f.write(markdown_table(depth_one_rows, ["dataset", "accuracy_mean", "accuracy_variance"]))
        f.write("\n\n## Max Depth Sweep, Information Gain\n\n")
        f.write(markdown_table(depth_grid_rows, ["dataset", "max_depth", "accuracy_mean", "accuracy_std"]))
        f.write("\n\n## Information Gain vs Gain Ratio\n\n")
        f.write(markdown_table(quality_rows, ["dataset", "max_depth", "quality_function", "accuracy_mean", "accuracy_variance"]))
        f.write("\n\n## Full-Training Accuracy\n\n")
        f.write(markdown_table(full_rows, ["dataset", "max_depth", "training_accuracy", "num_nodes"]))
        f.write("\n")

    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()