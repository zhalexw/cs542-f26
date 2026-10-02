"""Run the decision tree experiments from Tasks 5 and 7 of pa1.pdf.

Examples:
    python train_dt.py --task 5
    python train_dt.py --task 7
    python train_dt.py --task both
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.dt.clf import DecisionTreeClassifier
from src.dt.data.c45 import load_spam, load_volcanoes, load_voting
from src.dt.data.header import Header
from src.dt.quality.gr import GainRatio
from src.dt.quality.ig import InformationGain


DATASETS = {
    "voting": load_voting,
    "spam": load_spam,
    "volcanoes": load_volcanoes,
}
DEPTHS = (1, 2, 3, 4, 5)
COMPARISON_DEPTHS = (1, 3, 5)
ALPHAS = (0.5, 1.0, 1.5, 2.0, 2.5)
COMPARISON_ALPHAS = (0.5, 1.0, 2.0)
SEED = 12345
NUM_FOLDS = 5


def make_model(header: Header, quality_name: str) -> DecisionTreeClassifier:
    quality_function = InformationGain() if quality_name == "ig" else GainRatio()
    return DecisionTreeClassifier(header, quality_function)


def fit_at_depth(model: DecisionTreeClassifier, X: np.ndarray,
                 y_gt: np.ndarray, max_depth: int) -> None:
    # The root is at depth 1; max_depth=1 permits one split at the root.
    def stop_at_depth(_X, _y_gt, _feature_idxs, depth):
        return depth > max_depth

    model.fit(X, y_gt, pre_prune_function=stop_at_depth)


def make_folds(X: np.ndarray, y_gt: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True,
                               random_state=SEED)
    return list(splitter.split(X, y_gt))


def summarize_fold_accuracies(fold_accuracies: list[float]) -> dict:
    scores = np.asarray(fold_accuracies)
    result = {
        "mean_accuracy": float(np.mean(scores)),
        "variance": float(np.var(scores)),
        "std_dev": float(np.std(scores)),
    }
    result.update({f"fold_{idx}_accuracy": score
                   for idx, score in enumerate(fold_accuracies, start=1)})
    return result


def cross_validation_result(header: Header, X: np.ndarray, y_gt: np.ndarray,
                            folds: list[tuple[np.ndarray, np.ndarray]],
                            quality_name: str, max_depth: int) -> dict:
    fold_accuracies = []
    for train_idxs, test_idxs in folds:
        model = make_model(header, quality_name)
        fit_at_depth(model, X[train_idxs], y_gt[train_idxs], max_depth)
        y_hat = model.predict(X[test_idxs])
        fold_accuracies.append(float(np.mean(y_hat == y_gt[test_idxs])))

    return {"quality_function": quality_name, "max_depth": max_depth,
            **summarize_fold_accuracies(fold_accuracies)}


def pruning_cross_validation_result(
        header: Header, X: np.ndarray, y_gt: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
        quality_name: str, alpha: float) -> dict:
    fold_accuracies = []
    for train_idxs, test_idxs in folds:
        model = make_model(header, quality_name)
        model.fit(X[train_idxs], y_gt[train_idxs], mcc_prune=True, alpha=alpha)
        y_hat = model.predict(X[test_idxs])
        fold_accuracies.append(float(np.mean(y_hat == y_gt[test_idxs])))

    return {"quality_function": quality_name, "alpha": alpha,
            **summarize_fold_accuracies(fold_accuracies)}


def full_training_result(header: Header, X: np.ndarray, y_gt: np.ndarray,
                         max_depth: int) -> dict:
    model = make_model(header, "ig")
    fit_at_depth(model, X, y_gt, max_depth)
    return {
        "quality_function": "ig",
        "max_depth": max_depth,
        "training_accuracy": float(np.mean(model.predict(X) == y_gt)),
        "num_nodes": model.num_nodes,
    }


def pruning_full_training_result(header: Header, X: np.ndarray,
                                 y_gt: np.ndarray, alpha: float) -> dict:
    model = make_model(header, "ig")
    model.fit(X, y_gt, mcc_prune=True, alpha=alpha)
    return {
        "quality_function": "ig",
        "alpha": alpha,
        "training_accuracy": float(np.mean(model.predict(X) == y_gt)),
        "num_nodes": model.num_nodes,
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_depth_curve(dataset_name: str, rows: list[dict], output_dir: Path) -> None:
    ig_rows = sorted((row for row in rows
                      if row["dataset"] == dataset_name
                      and row["quality_function"] == "ig"),
                     key=lambda row: row["max_depth"])
    depths = [row["max_depth"] for row in ig_rows]
    means = [row["mean_accuracy"] for row in ig_rows]
    standard_deviations = [row["std_dev"] for row in ig_rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(depths, means, yerr=standard_deviations, fmt="o-", capsize=4)
    ax.set(title=f"{dataset_name.capitalize()}: 5-fold CV accuracy by max depth",
           xlabel="Maximum tree depth", ylabel="Mean classification accuracy")
    ax.set_xticks(depths)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset_name}_depth_accuracy.png", dpi=160)
    plt.close(fig)


def plot_alpha_curve(dataset_name: str, rows: list[dict], output_dir: Path) -> None:
    ig_rows = sorted((row for row in rows
                      if row["dataset"] == dataset_name
                      and row["quality_function"] == "ig"),
                     key=lambda row: row["alpha"])
    alphas = [row["alpha"] for row in ig_rows]
    means = [row["mean_accuracy"] for row in ig_rows]
    standard_deviations = [row["std_dev"] for row in ig_rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(alphas, means, yerr=standard_deviations, fmt="o-", capsize=4)
    ax.set(title=f"{dataset_name.capitalize()}: 5-fold CV accuracy by pruning alpha",
           xlabel="Pruning alpha", ylabel="Mean classification accuracy")
    ax.set_xticks(alphas)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{dataset_name}_alpha_accuracy.png", dpi=160)
    plt.close(fig)


def run_task5(dataset_names: list[str], output_dir: Path) -> None:
    cv_rows = []
    full_rows = []

    for dataset_name in dataset_names:
        header, X, y_gt = DATASETS[dataset_name]()
        folds = make_folds(X, y_gt)

        # Depth 2 supports the CV/full-data comparison for every dataset.
        ig_depths = DEPTHS if dataset_name in {"spam", "volcanoes"} \
            else sorted(set(COMPARISON_DEPTHS) | {2})
        for quality_name, depths in (("ig", ig_depths),
                                     ("gr", COMPARISON_DEPTHS)):
            for max_depth in depths:
                result = cross_validation_result(
                    header, X, y_gt, folds, quality_name, max_depth)
                cv_rows.append({"dataset": dataset_name, **result})
                print(f"{dataset_name} {quality_name} depth={max_depth}: "
                      f"mean={result['mean_accuracy']:.4f}, "
                      f"variance={result['variance']:.6f}", flush=True)

        for max_depth in (1, 2):
            result = full_training_result(header, X, y_gt, max_depth)
            full_rows.append({"dataset": dataset_name, **result})

    cv_fields = ["dataset", "quality_function", "max_depth", "mean_accuracy",
                 "variance", "std_dev"] + [
                     f"fold_{idx}_accuracy" for idx in range(1, NUM_FOLDS + 1)]
    write_csv(output_dir / "cross_validation.csv", cv_rows, cv_fields)
    write_csv(output_dir / "full_training.csv", full_rows,
              ["dataset", "quality_function", "max_depth",
               "training_accuracy", "num_nodes"])

    for dataset_name in ("spam", "volcanoes"):
        if dataset_name in dataset_names:
            plot_depth_curve(dataset_name, cv_rows, output_dir)


def run_task7(dataset_names: list[str], output_dir: Path) -> None:
    cv_rows = []
    full_rows = []

    for dataset_name in dataset_names:
        header, X, y_gt = DATASETS[dataset_name]()
        folds = make_folds(X, y_gt)

        ig_alphas = ALPHAS if dataset_name in {"spam", "volcanoes"} \
            else COMPARISON_ALPHAS
        for quality_name, alphas in (("ig", ig_alphas),
                                     ("gr", COMPARISON_ALPHAS)):
            for alpha in alphas:
                result = pruning_cross_validation_result(
                    header, X, y_gt, folds, quality_name, alpha)
                cv_rows.append({"dataset": dataset_name, **result})
                print(f"{dataset_name} {quality_name} alpha={alpha:g}: "
                      f"mean={result['mean_accuracy']:.4f}, "
                      f"variance={result['variance']:.6f}", flush=True)

        for alpha in (0.5, 1.0):
            result = pruning_full_training_result(header, X, y_gt, alpha)
            full_rows.append({"dataset": dataset_name, **result})

    cv_fields = ["dataset", "quality_function", "alpha", "mean_accuracy",
                 "variance", "std_dev"] + [
                     f"fold_{idx}_accuracy" for idx in range(1, NUM_FOLDS + 1)]
    write_csv(output_dir / "pruning_cross_validation.csv", cv_rows, cv_fields)
    write_csv(output_dir / "pruning_full_training.csv", full_rows,
              ["dataset", "quality_function", "alpha",
               "training_accuracy", "num_nodes"])

    for dataset_name in ("spam", "volcanoes"):
        if dataset_name in dataset_names:
            plot_alpha_curve(dataset_name, cv_rows, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("5", "7", "both"), default="5",
                        help="experiment suite to run (default: 5)")
    parser.add_argument("--output-dir", type=Path,
                        help="directory for CSV results and plots")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS), help="datasets to evaluate")
    args = parser.parse_args()

    output_dir = args.output_dir or Path(
        "decision_tree_results" if args.task == "both" else f"task{args.task}_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.task in {"5", "both"}:
        run_task5(args.datasets, output_dir)
    if args.task in {"7", "both"}:
        run_task7(args.datasets, output_dir)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
