# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from .activation import Activation


class Tanh(Activation):

    def forward(self: Tanh,
                X: np.ndarray) -> np.ndarray:
        return np.tanh(X)

    # also element-wise indepenent
    def backward(self: Tanh,
                 X: np.ndarray,
                 dLoss_dOutput: np.ndarray) -> np.ndarray:
        return dLoss_dOutput * (1 - (np.tanh(X) ** 2))

