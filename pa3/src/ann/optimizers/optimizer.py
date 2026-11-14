# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import numpy as np


# PYTHON PROJECT IMPORTS
from ..parameter import Parameter


class Optimizer(ABC):

    def __init__(self: Optimizer,
                 params: Sequence[Parameter],
                 lr: float) -> None:
        self.parameters: Sequence[Parameter] = params
        self.lr: float = lr

    def reset(self: Optimizer) -> None:
        # clear the gradients that have been accumulated in each Parameter object
        for P in self.parameters:
            P.reset()

    @abstractmethod
    def _step(self: Optimizer, P: Parameter) -> None:
        ...

    def step(self: Optimizer) -> None:
        for P in self.parameters:
            if not P.frozen:
                self._step(P)

