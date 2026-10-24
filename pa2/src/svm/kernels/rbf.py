# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS
from .linear import LinearKernel


class RBFKernel(LinearKernel):
    def __init__(self: RBFKernel, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma

    def sample_similarity(self: RBFKernel, x: np.ndarray, y: np.ndarray) -> float:
        return np.exp(-(super().sample_similarity(x, y)) / (2*(self.sigma**2)))

    def self_similarity(self: RBFKernel, X: np.ndarray) -> np.ndarray:
        return np.exp(-(super().self_similarity(X)) / (2*(self.sigma**2)))

    def batch_similarity(self: RBFKernel, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-(super().batch_similarity(X, y)) / (2*(self.sigma**2)))

    def __str__(self: RBFKernel) -> str:
        return f"RBFKernel(sigma={self.sigma})"

    def __repr__(self: RBFKernel) -> str:
        return str(self)

