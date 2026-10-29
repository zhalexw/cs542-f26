# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import scipy as sp


# PYTHON PROJECT IMPORTS
from .kernel import Kernel


class RBFKernel(Kernel):
    def __init__(self: RBFKernel, sigma: float = 1.0) -> None:
        super().__init__()
        self.sigma = sigma

    def sample_similarity(self: RBFKernel, x: np.ndarray, y: np.ndarray) -> float:
        return np.exp(-(np.linalg.norm(x-y, ord=2)**2) / (2*(self.sigma**2)))

    def self_similarity(self: RBFKernel, X: np.ndarray) -> np.ndarray:
        return np.exp(-(sp.spatial.distance.cdist(X, X, metric="euclidean")**2) / (2*self.sigma**2))

    def batch_similarity(self: RBFKernel, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.exp(-(np.linalg.norm(X-y.reshape(1,-1), ord=2, axis=1, keepdims=True)**2) / (2*(self.sigma**2)))

    def __str__(self: RBFKernel) -> str:
        return f"RBFKernel(sigma={self.sigma})"

    def __repr__(self: RBFKernel) -> str:
        return str(self)

