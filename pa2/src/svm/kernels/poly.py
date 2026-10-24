# SYSTEM IMPORTS
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS
from .linear import LinearKernel


class PolynomialKernel(LinearKernel):
    def __init__(self: PolynomialKernel, degree: int) -> None:
        super().__init__()
        self.degree = degree

    def sample_similarity(self: PolynomialKernel, x: np.ndarray, y: np.ndarray) -> float:
        return (super().sample_similarity(x, y)) ** self.degree

    def self_similarity(self: PolynomialKernel, X: np.ndarray) -> np.ndarray:
        return (super().self_similarity(X)) ** self.degree

    def batch_similarity(self: PolynomialKernel, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (super().batch_similarity(X, y)) ** self.degree

    def __str__(self: PolynomialKernel) -> str:
        return f"PolynomialKernel(degree={self.degree})"

    def __repr__(self: PolynomialKernel) -> str:
        return str(self)


class QuadradicKernel(PolynomialKernel):
    def __init__(self: QuadradicKernel) -> None:
        super().__init__(2)

    def __str__(self: QuadradicKernel) -> str:
        return f"QuadradicKernel()"

    def __repr__(self: QuadradicKernel) -> str:
        return str(self)

