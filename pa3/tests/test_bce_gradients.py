# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence, Set
from tqdm import tqdm
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
from src.ann.losses.lossfunction import LossFunction
from src.ann.model import Model
from src.ann.optimizers.adam import AdamOptimizer
from src.ann.optimizers.optimizer import Optimizer
from src.ann.optimizers.sgd import SGDOptimizer
from src.ann.parameter import Parameter



# numerical gradient checking...this is how we check whether
# our backpropogation works or not. What we do is compute the
# numerical partial derivative with respect to each learnable
# parameter. A numerical derivative can be computed as:
#       df/dx = (f(x+e) - f(x-e))/(2*e)
# if we set "e" to be small, then we can approximate
# the gradient. We can compare the symbolic gradients
# versus the numerical gradients, and hope they are super close.
#
# Note that in our loss functions, the constant DELTA will throw off
# these measurements (but it is worth it for numerical stability)
# so we need to relax our tolerances a bit (which is why they're so high)
def grad_check(X: np.ndarray,
               Y_gt: np.ndarray,
               model: ANNC,
               lf: LossFunction,
               epsilon: float = 1e-5,           # value 'e': how much to poke a parameter
               delta: float = 1e-4) -> None:    # tolerance for error between symbolic and numerical grads
    params: List[Parameter] = [P for P in model.parameters() if not P.frozen]   # only grab non-frozen params
    num_grads: List[np.ndarray] = [np.zeros_like(P.val) for P in params]
    sym_grads: List[np.ndarray] = [P.grad for P in params]

    for P, NG in zip(params, num_grads):
        for index, v in np.ndenumerate(P.val):
            P.val[index] += epsilon
            NG[index] += lf.forward(model._predict(X), Y_gt)

            P.val[index] -= 2*epsilon
            NG[index] -= lf.forward(model._predict(X), Y_gt)

            # set param back to normal
            P.val[index] = v
            NG[index] /= (2*epsilon)

    ratios: np.ndarray = np.array([np.linalg.norm(SG-NG) / np.linalg.norm(SG+NG)
                                   for SG, NG in zip(sym_grads, num_grads)], dtype=float)
    if np.sum(ratios > delta) > 0:
        raise RuntimeError("ERROR: failed grad check. delta: [%s], ratios: %s"
            % (delta, ratios))



class TestBCEGradients(unittest.TestCase):

    def test_sgd_bce_gradients(self: TestBCEGradients) -> None:
        np.random.seed(12345)
        NUM_EPOCHS: int = 1000

                                           # all discrete       # all continuous        # TODO: a mix?
        for data_name, data_loader in zip(["play_outside",      "small_linsep"],
                                          [load_play_outside,   load_small_linsep]):
            header, X_user_samples, y_gt = data_loader()

            # skip ReLU(): its numerical gradients aren't very stable around the discontinuity
            for hidden_activation_function in [Linear(), Sigmoid(), Tanh()]:
                for lr in [1e-6, 1e-5, 1e-4, 1e-3]:
                    model = ANNC(header, 10, hidden_activation_function)

                    # map the data and cache for speed
                    X_network_samples: np.ndarray = model._map_user_samples(X_user_samples)
                    Y_gt: np.ndarray = y_gt.reshape(-1,1)

                    # train for a bit
                    optimizer = SGDOptimizer(model.parameters(), lr)
                    for _ in tqdm(range(NUM_EPOCHS),
                                  desc=f"checking gradients for ANNC(n_hidden={10},"
                                    + f" activation={hidden_activation_function.__class__.__name__})"
                                    + f" w/ lr={lr} on data={data_name})"):
                        
                        optimizer.reset()
                        Y_hat = model._predict(X_network_samples)
                        model._backward(X_network_samples, Y_gt, gamma=0.0) # important to set gamma=0 for this

                        # grad check
                        grad_check(X_network_samples, Y_gt, model, model.loss_function)

                        # take a gradient descent step
                        optimizer.step()
                        

if __name__ == "__main__":
    unittest.main()

