# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from .lossfunction import LossFunction


DELTA: float = 1e-12 # numerical stability


class BinaryCrossEntropy(LossFunction):

    def forward(self: BinaryCrossEntropy,
                Y_hat: np.ndarray,
                Y_gt: np.ndarray) -> float:
        assert(Y_hat.shape == Y_gt.shape)
        return (np.sum(-np.log(Y_hat[Y_gt!=0]+DELTA)) +\
               np.sum(-np.log(1-Y_hat[Y_gt==0]+DELTA)))/(Y_hat.shape[0])

    # take the derivative of binary cross entropy with respect to Y_hat
    def backward(self: BinaryCrossEntropy,
                 Y_hat: np.ndarray,
                 Y_gt: np.ndarray) -> np.ndarray:
        assert(Y_hat.shape == Y_gt.shape)

        dL_dYhat = np.zeros_like(Y_hat)
        dL_dYhat[Y_gt!=0] += -1.0/(Y_hat[Y_gt!=0]+DELTA)
        dL_dYhat[Y_gt==0] += 1.0/(1-Y_hat[Y_gt==0]+DELTA)

        dL_dYhat /= Y_hat.shape[0]
        return dL_dYhat

