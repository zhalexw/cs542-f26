# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS
from .kernel import Kernel


class LinearKernel(Kernel):
    def __init__(self: LinearKernel) -> None:
        ...

    def sample_similarity(self: LinearKernel, x: np.ndarray, y: np.ndarray) -> float:
        return np.dot(x.reshape(-1), y.reshape(-1))

    def self_similarity(self: LinearKernel, X: np.ndarray) -> np.ndarray:
        return np.dot(X, X.T)

    def batch_similarity(self: LinearKernel, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.dot(X, y.reshape(-1, 1))

    def __str__(self: NoKernel) -> str:
        return "LinearKernel()"

    def __repr__(self: LinearKernel) -> str:
        return str(self)

