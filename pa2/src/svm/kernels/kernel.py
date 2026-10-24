# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS


class Kernel(ABC):

    @abstractmethod
    def sample_similarity(self: Kernel, x: np.ndarray, y: np.ndarray) -> float:
        ...

    @abstractmethod
    def self_similarity(self: Kernel, X: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def batch_similarity(self: NoKernel, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ...

