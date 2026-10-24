# SYSTEM IMPORTS
from __future__ import annotations
from multiprocessing import Pool
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
from src.svm.clf import SVMC
from src.svm.data.c45 import load_play_outside, load_small_linsep
from src.svm.kernels.poly import QuadradicKernel
from src.svm.kernels.rbf import RBFKernel
from src.svm.kernels.linear import LinearKernel


def train_return_params(data) -> tuple[np.ndarray, float]:
    header, X, y_gt = data
    m = SVMC(header, LinearKernel())
    m.fit(X, y_gt, max_iters=10000, C=1000, epsilon=0.001, tol=1e-5) # large C for linearly separable
    return m.w, m.b


class TestSVMCLinearlySeparableData(unittest.TestCase):

    def test_w_b_play_outside(self: TestSVMCLinearlySeparableData) -> None:
        num_models = 200
        num_cpus_cores = max(1, os.cpu_count()-2) # change this for your degree of parallelization

        # going to average 'num_models' models on this data each using max_iters=10000
        params = None
        with Pool(num_cpus_cores) as pool: # keep 2 cores freed up
            params = pool.map(train_return_params, [load_play_outside() for _ in range(num_models)]) 

        w_avg = np.zeros(4)
        b_avg = 0

        for w, b in params:
            w_avg += w
            b_avg += b

        w_avg /= num_models
        b_avg /= num_models

        w_avg_expected = np.array([-3.99405514, 2.99733626, 7.99251494, -5.99251494])
        b_avg_expected = 0.9934377492117341

        np.testing.assert_allclose(w_avg, w_avg_expected, atol=1e-3,
            err_msg="When running my solution I got this average across 200 models")
        self.assertAlmostEqual(b, b_avg_expected, delta=1e-2,
            msg="When running my solution I got this average across 200 models")

    def test_w_b_small_linsep(self: TestSVMCLinearlySeparableData) -> None:
        num_models = 500
        num_cpus_cores = max(1, os.cpu_count()-2) # change this for your degree of parallelization

        # going to average 'num_models' models on this data each using max_iters=10000
        params = None
        with Pool(num_cpus_cores) as pool: # keep 2 cores freed up
            params = pool.map(train_return_params, [load_small_linsep() for _ in range(num_models)]) 

        w_avg = np.zeros(2)
        b_avg = 0

        for w, b in params:
            w_avg += w
            b_avg += b

        w_avg /= num_models
        b_avg /= num_models

        # the optimal hyperplane w/ a linear kernel (no feature warping)
        w_avg_expected = np.array([-0.5, 0.5])
        b_avg_expected = 0

        np.testing.assert_allclose(w_avg, w_avg_expected, atol=1e-3,
            err_msg="the optimal hyperplane has w=[-0.5, +0.5]")
        self.assertAlmostEqual(b, b_avg_expected, delta=1e-4,
            msg="the optimal hyperplane has b=0")


if __name__ == "__main__":
    unittest.main()

