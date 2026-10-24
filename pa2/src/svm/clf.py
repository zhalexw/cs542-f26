# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence, Set
from typing import Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS
from .data.header import Feature, FeatureType, ContinuousFeature, DiscreteFeature, Header
from .kernels.kernel import Kernel
from .model import Model
from .opt.smo import smo


"""
    A Support Vector Machine Classifier (SVMC) model. This model exclusively performs binary classification
    where the ground truth is either +1 or -1. When data is presented to this model, the classes of the binary
    classification problem may or not adhere to the {-1,+1} label requirement, so this model internally will
    map "actual classes" (e.g. class labels the data is originally expressed with) to "expected classes"
    (e.g. {-1,+1} that the SVMC expects). When data is presented to the model the model will internally perform
    this mapping to get labels useful for training the SVMC, and during inference will map from {-1, +1} back to
    the "actual classes" that whoever is using the SVMC can work with.
"""
class SVMC(Model):
    def __init__(self: SVMC,
                 header: Header,
                 kernel: Kernel) -> None:
        self.header: Header = header
        self.kernel = kernel

        # support vector machines are designed exclusively where the class labels are +1 and -1
        # but the input data may or may not adhere to that definition. So you will have to map
        # one class to +1 and another class to -1
        self.class_to_expected_class, self.expected_class_to_class = self._make_class_maps(header)

        self.w: np.ndarray = None
        self.b: float = None

    def _make_class_maps(self: SVMC, header: Header) -> tuple[Dict[int, int], Dict[int, int]]:
        num_classes: int = len(header.ground_truth.domain())
        if num_classes != 2:
            raise ValueError(f"[ERROR]: cannot train a SVMC on non-binary data: found {num_classes} classes!")

        class_to_expected_class: Dict[int, int] = dict()
        expected_class_to_class: Dict[int, int] = dict()

        # map unique values to -1, +1
        for expected_class, raw_class_value in zip([-1, +1], sorted(header.ground_truth.domain())):
            class_idx = header.ground_truth.map(raw_class_value)

            class_to_expected_class[class_idx] = expected_class
            expected_class_to_class[expected_class] = class_idx

        return class_to_expected_class, expected_class_to_class

    def _apply_class_map(self: SVMC, class_map: Dict[int, int], y: np.ndarray) -> np.ndarray:
        y_out: np.ndarray = np.zeros_like(y.reshape(-1))
        for idx, class_idx in enumerate(y.reshape(-1)):
            y_out[idx] = class_map[class_idx]
        return y_out.reshape(*y.shape)

    def _to_expected_classes(self: SVMC, y_gt: np.ndarray) -> np.ndarray:
        return self._apply_class_map(self.class_to_expected_class, y_gt)

    def _to_actual_classes(self: SVMC, y_hat: np.ndarray) -> np.ndarray:
        return self._apply_class_map(self.expected_class_to_class, y_hat)

    def _to_w(self: SVMC, alphas: np.ndarray, X: np.ndarray, y_gt: np.ndarray) -> np.ndarray:

        n = len(y_gt)
        w = np.zeros(X.shape[1])

        for i in range(n):
            w += alphas[i]*y_gt[i]*X[i]

        return w

        
    def fit(self: SVMC,
            X: np.ndarray,
            y_gt: np.ndarray,
            C: float = 1.0,
            epsilon: float = 1e-8, tol: float=1e-5,
            max_iters: int = int(1e6)) -> None:

        # make sure that the ground truth has elements {-1, +1}
        y_gt = self._to_expected_classes(y_gt)

        alphas, b = smo(self.kernel, X, y_gt, C=C, epsilon=epsilon, tol=tol, max_iters=max_iters)

        self.w = self._to_w(alphas, X, y_gt)
        self.b = b

        return None

    def predict(self: SVMC,
                X: np.ndarray) -> np.ndarray:
        """
            TODO: return a 1d array of class labels! You should
                - calculate the signed margins using the learned hyperplane for each test point
                - get the sign of each signed margin. Be careful if you choose to use np.sign, this function
                  will assign a value of 0 to 0 values. We want either +1 or -1 with ties going to +1
                - map the {-1, +1} labels back to the actual class lables used by the caller of this method.
        """

        w = self.w
        b = self.b

        margins = X @ w + b
        y_hat = np.where(margins >= 0, 1, -1)

        return self._to_actual_classes(y_hat)
        

    def __str__(self: SVMC) -> str:
        return f"SVMC(header={self.header}, kernel={self.kernel}, w={self.w}, b={self.b})"

    def __repr__(self: SVMC) -> str:
        return str(self)


"""
    This is a "One-Vs-[The] Rest" model. In order to perform multiclass classification using classifiers
    which can *only* perform binary classification, you need to build an ensemble of binary classifiers
    which break down the multiclass problem into (many) binary classification problems.

    As discussed in lecture, the "One-Vs-Rest" paradigm breaks up a mulitclass classification problem into
    a linear number of binary classification problems: each binary classification problem focusing on
        - "is this sample class 'i' or not?"
"""
class OVRC(Model):
    def __init__(self: OVRC,
                 header: Header,
                 kernel: Kernel) -> None:
        self.header = header
        self.kernel = kernel

        self.constituent_svmcs: Sequence[SVMC] = list()

    def fit(self: OVRC,
            X: np.ndarray,
            y_gt: np.ndarray,
            C: float = 1.0,
            epsilon: float = 1e-9,
            max_iters: int = int(1e6)) -> None:
        """
            TODO: train each constituent model on the training data. This may involve messing with the training
                  data somehow!
        """
        ...

    def predict(self: OVRC,
                X: np.ndarray) -> np.ndarray:
        """
            TODO: ask each constituent model to predict which class it thinks it is and combine them together
                  somehow to infer the overall class your ensemble should predict!
        """
        ...

    def __str__(self: OVRC) -> str:
        return f"OVRC(header={self.header}, kernel={self.kernel}, svmcs={self.svmcs})"

    def __repr__(self: OVRC) -> str:
        return str(self)


"""
    This is a "One-Vs-One" model. In order to perform multiclass classification using classifiers
    which can *only* perform binary classification, you need to build an ensemble of binary classifiers
    which break down the multiclass problem into (many) binary classification problems.

    As discussed in lecture, the "One-Vs-One" paradigm breaks up a mulitclass classification problem into
    a quadratic number of binary classification problems: each binary classification problem focusing on
        - "is this sample class 'i' or class 'j'?"
"""
class OVOC(Model):
    def __init__(self: OVOC,
                 header: Header,
                 kernel: Kernel) -> None:
        self.header = header
        self.kernel = kernel

        self.constituent_svmcs: Mapping[tuple[int, int], SVMC] = dict()

    def fit(self: OVOC,
            X: np.ndarray,
            y_gt: np.ndarray,
            C: float = 1.0,
            epsilon: float = 1e-9,
            max_iters: int = int(1e6)) -> None:
        """
            TODO: train each constituent model on the training data. This may involve messing with the training
                  data somehow!
        """
        ...

    def predict(self: OVOC,
                X: np.ndarray) -> np.ndarray:
        """
            TODO: ask each constituent model to predict which class it thinks it is and combine them together
                  somehow to infer the overall class your ensemble should predict!
        """
        ...

    def __str__(self: OVOC) -> str:
        return f"OVOC(header={self.header}, kernel={self.kernel}, svmcs={self.svmcs})"

    def __repr__(self: OVOC) -> str:
        return str(self)

