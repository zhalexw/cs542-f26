# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from .activation import Activation


class Linear(Activation):

    def forward(self: Linear,
                X: np.ndarray) -> np.ndarray:
        return X

    # also element-wise indepenent
    def backward(self: Linear,
                 X: np.ndarray,
                 dLoss_dOutput: np.ndarray) -> np.ndarray:
        return dLoss_dOutput

