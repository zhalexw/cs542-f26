# SYSTEM IMPORTS
from __future__ import annotations
import argparse as ap
import numpy as np
import os
import sys
import unittest

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


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
    if args == "quad":
        kernel = QuadradicKernel()
    elif args == "rbf":
        kernel = RBFKernel()
    elif args == "linear":
        kernel = LinearKernel()

    return SVMC(header, kernel)

def make_folds(X, y_gt):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
    return list(splitter.split(X, y_gt))


def main() -> None:
    header, X, y_gt = load_data("play_outside")
    t = make_folds(X, y_gt)
    for train, test in t:
        print(train)
        print(test)
    

    for dataset in ["spam", "volcanoes", "voting"]:
        header, X, y_gt = load_data(dataset)
        folds = make_folds(X, y_gt)
        for kernel in ["linear", "rbf", "quad"]:
            model = make_model(header, kernel)
            for c in [1000, 500, 100, 10, 1]:
                if dataset == "voting" and (c == 500 or c == 10):
                    continue
                for train_idxs, test_idxs in folds:
                    model.fit(X[train_idxs], y_gt[train_idxs], C=c, max_iters=100000, epsilon=1e-08, tol=1e-05)
                    print(model)

                    # get the accuracy of the model
                    print(f"{dataset} {kernel} accuracy C={c}", "=>", np.sum(y_gt[test_idxs] == model.predict(X[test_idxs]).reshape(-1)) / y_gt[test_idxs].shape[0])


    
    
    
    """# fit scikit-learns implementation and measure its accuracy
    m = SVC(kernel=args.kernel if "rbf" in args.kernel or "linear" in args.kernel else "linear",
            C=args.penalty_coeff, max_iter=args.max_iters, tol=args.tol)
    m.fit(X, y_gt)
    print("sklearn accuracy", "=>", np.sum(y_gt[test_idxs] == m.predict(X[test_idxs]).reshape(-1)) / y_gt[test_idxs].shape[0])"""


if __name__ == "__main__":
    main()

