# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from .activation import Activation


class Tanh(Activation):

    def forward(self: Tanh,
                X: np.ndarray) -> np.ndarray:
        # TODO
        ...

    # also element-wise indepenent
    def backward(self: Tanh,
                 X: np.ndarray,
                 dLoss_dOutput: np.ndarray) -> np.ndarray:
        # TODO
        ...

