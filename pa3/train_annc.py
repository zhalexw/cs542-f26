# SYSTEM IMPORTS
from __future__ import annotations
import argparse as ap
import numpy as np
import os
import sys
import unittest
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_digits, load_iris


# can't do relative file importing since this is an executable
# so need to ensure required base paths are on sys.path
_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from src.ann.clf import ANNC
from src.ann.data.c45 import *
from src.ann.data.header import Header, ContinuousFeature, GroundTruth, FeatureType
from src.ann.activations.activation import Activation
from src.ann.activations.linear import Linear
from src.ann.activations.relu import ReLU
from src.ann.activations.sigmoid import Sigmoid
from src.ann.activations.softmax import Softmax
from src.ann.activations.tanh import Tanh
from src.ann.model import Model
from src.ann.optimizers.optimizer import Optimizer
from src.ann.optimizers.sgd import SGDOptimizer

from sklearn.preprocessing import StandardScaler

def load_data(dataset_name: str) -> tuple[Header, np.ndarray, np.ndarray]:
    if dataset_name == "play_outside":
        return load_play_outside()
    elif dataset_name == "spam":
        return load_spam()
    elif dataset_name == "volcanoes":
        return load_volcanoes()
    elif dataset_name == "voting":
        return load_voting()
    else:
        # TODO: add more datasets if you want
        return None, None, None


def make_hidden_activation(args) -> Activation:
    activation = None
    if args.hidden_activation == "linear":
        activation = Linear()
    elif args.hidden_activation == "relu":
        activation = ReLU()
    elif args.hidden_activation == "sigmoid":
        activation = Sigmoid()
    elif args.hidden_activation == "tanh":
        activation = Tanh()
    elif args.hidden_activation == "softmax":
        activation = Softmax()
    else:
        # TODO: add more activation functions if you want
        ...
    return activation


def make_model(header: Header, args) -> Model:
    hidden_activation = make_hidden_activation(args)

    return ANNC(header, args.num_hidden_units, hidden_activation)


def make_loss_function(header: Header, args) -> LossFunction:
    return BinaryCrossEntroy() if len(header.ground_truth.domain) == 2 else CategoricalCrossEntropy()



def make_folds(X, y):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
    return list(splitter.split(X, y))


def evaluate_fold(header, X, y, train_idx, test_idx, fold_number, args):
    np.random.seed(12345 + fold_number)
    X_train = X[train_idx].astype(float, copy=True)
    X_test = X[test_idx].astype(float, copy=True)

    continuous_columns = [column for column, feature in enumerate(header.features) if feature.type == FeatureType.CONTINUOUS]

    if continuous_columns:
        scaler = StandardScaler()
        X_train[:, continuous_columns] = scaler.fit_transform(X_train[:, continuous_columns])
        X_test[:, continuous_columns] = scaler.transform(X_test[:, continuous_columns])


    model = make_model(header, args)
    optimizer = SGDOptimizer(model.parameters(), args.lr)
    #model.fit(X[train_idx], y[train_idx], optimizer, max_iters=args.max_iters, epsilon=args.epsilon, gamma=args.gamma)

    model.fit(X_train, y[train_idx], optimizer, max_iters = args.max_iters, epsilon = args.epsilon, gamma = args.gamma)

    y_test = y[test_idx]
    #scores = model.predict_proba(X[test_idx])
    scores = model.predict_proba(X_test)
    if scores.shape[1] == 1:
        scores = scores.reshape(-1)
        predictions = scores >= 0.5
        aroc = roc_auc_score(y_test, scores)
        average = "binary"
    else:
        predictions = scores.argmax(axis=1)
        aroc = roc_auc_score(y_test, scores, multi_class="ovr", average="weighted")
        average = "weighted"
    return {
        "AROC": aroc,
        "Precision": precision_score(y_test, predictions, average=average, zero_division=0),
        "Recall": recall_score(y_test, predictions, average=average, zero_division=0),
        "Accuracy": accuracy_score(y_test, predictions),
    }


def cross_validate(header, X, y, folds, args, jobs):
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return Parallel(n_jobs=jobs)(
            delayed(evaluate_fold)(header, X, y, train_idx, test_idx, fold_number, args)
            for fold_number, (train_idx, test_idx) in enumerate(folds)
        )

def make_header(dataset):
    header = Header()
    for col, name in enumerate(dataset.feature_names):
        header.add_feature(ContinuousFeature(name, col))
    header.add_ground_truth(GroundTruth([str(name) for name in dataset.target_names]))
    return header


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("dataset", choices=["play_outside", "spam", "volcanoes", "voting", "digits", "iris"],
                        help="dataset name to load")
    parser.add_argument("num_hidden_units", type=int, help="number of artificial neurons in the single hidden layer")
    parser.add_argument("hidden_activation", choices=["linear", "relu", "sigmoid", "tanh", "softmax"],
                        help="which kind of kernel function to use")
    parser.add_argument("-m", "--max_iters", type=int, default=10000, help="max number of iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9, help="convergence threshold")
    parser.add_argument("-n", "--lr", type=float, default=0.0001, help="gradient descent learning rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.01, help="weight decay coeff")
    parser.add_argument("--jobs", type=int, default=5, choices=range(1, 6),
                        help="number of parallel cross-validation workers (default: 5)")
    args = parser.parse_args()

    if args.dataset in ("digits", "iris"):
        dataset = load_digits() if args.dataset == "digits" else load_iris()
        header = make_header(dataset)
        X, y_gt = dataset.data, dataset.target
    else:
        header, X, y_gt = load_data(args.dataset)

    folds = make_folds(X, y_gt)
    results = cross_validate(header, X, y_gt, folds, args, args.jobs)
    print(f"{'Metric':<12} {'Fold 1':>9} {'Fold 2':>9} {'Fold 3':>9} "
          f"{'Fold 4':>9} {'Fold 5':>9} {'Mean':>9} {'Variance':>12}")
    for metric in results[0]:
        values = np.array([fold[metric] for fold in results])
        fold_scores = " ".join(f"{value:9.4f}" for value in values)
        print(f"{metric:<12} {fold_scores} {values.mean():9.4f} {values.var():12.6f}")


if __name__ == "__main__":
    main()

# linear relu sigmoid tanh
# 100 1000 10000
# volcanoes 225
