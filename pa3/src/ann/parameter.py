# SYSTEM IMPORTS
from __future__ import annotations
import numpy as np


# PYTHON PROJECT IMPORTS


# a matrix (or vector) containing a grouping of parameters called the 'value' matrix/vector
# (for instance a weight matrix or a bias vector for a specific layer)
#
# In our implementation a Parameter object will hold the matrix (or vector)
# as well as the current gradient for that matrix (or vector). Performing backprop
# is all out setting this gradient matrix/vector correctly.
#
# The Parameter type is also responsible for incrementing the 'value' matrix/vector with
# its gradient matrix/vector. It is on the caller to make sure that the 'gradient' matrix/vector
# is currently containing the values they want the parameter 'value' to be incremented by.
class Parameter(object):
    def __init__(self: Parameter,
                 V: np.ndarray,
                 frozen: bool = False) -> None:
        self.val: np.ndarray = V
        self.grad: np.ndarray = None

        # is this parameter allowed to change?
        self.frozen: bool = frozen

    def freeze(self: Parameter) -> None:
        self.frozen = True

    def thaw(self: Parameter) -> None:
        self.frozen = False

    def reset(self: Parameter) -> Parameter:
        self.grad = np.zeros_like(self.val)
        return self

    def step(self: Parameter,
             G: np.ndarray) -> Parameter:
        self.val -= G # np.clip(G, -100, 100)
        return self.reset()

