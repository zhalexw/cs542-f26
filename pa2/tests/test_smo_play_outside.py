# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence, Set
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
from src.svm.data.c45 import load_play_outside
from src.svm.kernels.poly import QuadradicKernel
from src.svm.kernels.rbf import RBFKernel
from src.svm.kernels.linear import LinearKernel
from src.svm.opt.smo import smo, _get_errors, _smo_step, _alphas_violating_kkt, _choose_alpha_violating_kkt


class TestSMOPlayOutside(unittest.TestCase):

    # load the data only once...loading data in each test is unnecessary and slow
    @classmethod
    def setUpClass(cls) -> None:
        header, X, y_gt = load_play_outside()
        cls.header_play_outside = header
        cls.X_play_outside = X
        cls.y_gt_play_outside = y_gt

        # map ground truth to +1, -1
        cls.y_gt_play_outside[cls.y_gt_play_outside == 0] = -1
        cls.y_gt_play_outside[cls.y_gt_play_outside == 1] = +1

    def test_smo__get_errors(self: TestSMOPlayOutside) -> None:
        for kernel_type, correct_errors in zip([QuadradicKernel, RBFKernel, LinearKernel],
                                               [np.array([1.,1.,-1.,-1.,-1.,1.,-1.,1.,-1.,-1.,-1.,-1.,-1.,1.]),
                                                np.array([1.,1.,-1.,-1.,-1.,1.,-1.,1.,-1.,-1.,-1.,-1.,-1.,1.]),
                                                np.array([1.,1.,-1.,-1.,-1.,1.,-1.,1.,-1.,-1.,-1.,-1.,-1.,1.])]):
            K = kernel_type()
            alphas: np.ndarray = np.zeros(self.X_play_outside.shape[0])
            b: float = 0

            # measure the errors
            errors: np.ndarray = _get_errors(K, alphas, b, self.X_play_outside, self.y_gt_play_outside)
            np.testing.assert_allclose(errors, correct_errors,
                                       err_msg="When I ran my solution on these inputs I got these errors")


    def test_smo_smo(self: TestSMOPlayOutside) -> None:

        # I found that with these kernels and max_iters=1000 I get stable results from smo
        for kernel_type, correct_alphas, correct_b in zip([RBFKernel, LinearKernel],
                                                          [np.array([1.,0.,0.,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]),
                                                           np.array([1.00000000e+00,7.31735671e-06,1.99976376e-01,
                                                                     1.00000000e+00,2.00013310e-01,1.00000000e+00,
                                                                     0.00000000e+00,1.00000000e+00,1.00000000e+00,
                                                                     0.00000000e+00,1.00000000e+00,6.00017631e-01,
                                                                     0.00000000e+00,1.00000000e+00])],
                                                          [-1.026625142306357, 0.5999880144311618]):
            K = kernel_type()
            alphas: np.ndarray = np.zeros(self.X_play_outside.shape[0])
            b: float = 0

            alphas, b = smo(K, self.X_play_outside, self.y_gt_play_outside, max_iters=1000)
            np.testing.assert_allclose(alphas, correct_alphas, atol=1e-4,
                                       err_msg="When I ran my solution on these inputs I got these stable results")
            self.assertAlmostEqual(b, correct_b, delta=1e-4,
                                   msg="When I ran my solutin on these input I got these stable results")

if __name__ == "__main__":
    unittest.main()

