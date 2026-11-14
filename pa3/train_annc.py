# SYSTEM IMPORTS
from __future__ import annotations
import argparse as ap
import numpy as np
import os
import sys
import unittest


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
from src.ann.data.header import Header
from src.ann.activations.activation import Activation
from src.ann.activations.linear import Linear
from src.ann.activations.relu import ReLU
from src.ann.activations.sigmoid import Sigmoid
from src.ann.activations.softmax import Softmax
from src.ann.activations.tanh import Tanh
from src.ann.model import Model
from src.ann.optimizers.optimizer import Optimizer
from src.ann.optimizers.sgd import SGDOptimizer


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


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("dataset", choices=["play_outside", "spam", "volcanoes", "voting"],
                        help="dataset name to load")
    parser.add_argument("num_hidden_units", type=int, help="number of artificial neurons in the single hidden layer")
    parser.add_argument("hidden_activation", choices=["linear", "relu", "sigmoid", "tanh", "softmax"],
                        help="which kind of kernel function to use")
    parser.add_argument("-m", "--max_iters", type=int, default=int(1e6), help="max number of iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=1e-9, help="convergence threshold")
    parser.add_argument("-n", "--lr", type=float, default=1e-4, help="gradient descent learning rate")
    parser.add_argument("-g", "--gamma", type=float, default=0.0, help="weight decay coeff")
    args = parser.parse_args()

    header, X, y_gt = load_data(args.dataset)
    model = make_model(header, args)

    # now we need to initialize our optimizer
    optimizer = SGDOptimizer(model.parameters(), args.lr)

    model.fit(X, y_gt, optimizer, max_iters=args.max_iters, epsilon=args.epsilon, gamma=args.gamma)
    print(model)

    # get the accuracy of the model
    y_hat: np.ndarray = model.predict(X).reshape(-1)
    print(y_hat, y_gt)
    print("ANN accuracy", "=>", np.sum(y_gt == y_hat) / y_gt.shape[0])


if __name__ == "__main__":
    main()

