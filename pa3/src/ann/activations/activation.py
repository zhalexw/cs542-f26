# SYSTEM IMPORTS
from __future__ import annotations
from abc import abstractmethod, ABC
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS


class Activation(ABC):

    @abstractmethod
    def forward(self: Module,
                X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self: Module,
                 X: np.ndarray,
                 dLoss_dOutput: np.ndarray) -> np.ndarray:
        ...

