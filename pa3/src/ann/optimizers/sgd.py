# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS
from .optimizer import Optimizer
from ..parameter import Parameter


class SGDOptimizer(Optimizer):

    def _step(self: SGDOptimizer,
              P: Parameter) -> None:
        P.step(self.lr * P.grad)

