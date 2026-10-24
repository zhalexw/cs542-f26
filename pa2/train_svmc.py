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
from src.svm.clf import SVMC, OVRC, OVOC
from src.svm.data.c45 import *
from src.svm.data.header import Header
from src.svm.kernels.poly import QuadradicKernel
from src.svm.kernels.rbf import RBFKernel
from src.svm.kernels.linear import LinearKernel
from src.svm.model import Model

from sklearn.svm import SVC


def load_data(dataset_name: str) -> tuple[Header, np.ndarray, np.ndarray]:
    if dataset_name == "play_outside":
        return load_play_outside()
    elif dataset_name == "spam":
        return load_spam()
    elif dataset_name == "volcanoes":
        return load_volcanoes()
    elif dataset_name == "voting":
        return load_voting()
    elif dataset_name == "small_linsep":
        return load_small_linsep()
    else:
        # TODO: add more datasets if you want
        return None, None, None


def make_model(header: Header, args) -> Model:
    kernel = None
    if args.kernel == "quad":
        kernel = QuadradicKernel()
    elif args.kernel == "rbf":
        kernel = RBFKernel()
    elif args.kernel == "linear":
        kernel = LinearKernel()
    else:
        # TODO: add more quality functions if you want
        ...

    return SVMC(header, kernel)


def main() -> None:
    parser = ap.ArgumentParser()
    parser.add_argument("dataset", choices=["play_outside", "spam", "volcanoes", "voting", "small_linsep"],
                        help="dataset name to load")
    parser.add_argument("kernel", choices=["quad", "rbf", "linear"], help="which kind of kernel function to use")
    parser.add_argument("-c", "--penalty_coeff", type=float, default=1.0, help="value of C")
    parser.add_argument("-m", "--max_iters", type=int, default=100, help="max number of iterations")
    parser.add_argument("-e", "--epsilon", type=float, default=0.001, help="max tolerance for deviations in alphas")
    parser.add_argument("-t", "--tol", type=float, default=1e-5, help="max tolerance for KKT conditions")
    args = parser.parse_args()

    header, X, y_gt = load_data(args.dataset)

    model = make_model(header, args)

    model.fit(X, y_gt, C=args.penalty_coeff, max_iters=args.max_iters, epsilon=args.epsilon, tol=args.tol)
    print(model)

    # get the accuracy of the model
    print("SVMC accuracy", "=>", np.sum(y_gt == model.predict(X).reshape(-1)) / y_gt.shape[0])

    # fit scikit-learns implementation and measure its accuracy
    m = SVC(kernel=args.kernel if "rbf" in args.kernel or "linear" in args.kernel else "linear",
            C=args.penalty_coeff, max_iter=args.max_iters, tol=args.tol)
    m.fit(X, y_gt)
    print("sklearn accuracy", "=>", np.sum(y_gt == m.predict(X).reshape(-1)) / y_gt.shape[0])


if __name__ == "__main__":
    main()

