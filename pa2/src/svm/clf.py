# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence, Set
from typing import Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS
from .data.header import Feature, FeatureType, ContinuousFeature, DiscreteFeature, Header, GroundTruth
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

        self.sv_coeffs: np.ndarray = None
        self.X_svs: np.ndarray = None
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

    def _set_svs(self: SVMC, alphas: np.ndarray, X: np.ndarray, y_gt: np.ndarray, epsilon: float) -> None:
        alphas = alphas.reshape(-1)
        y_gt = y_gt.reshape(-1)

        support_mask = np.abs(alphas) > epsilon
        self.sv_coeffs = alphas[support_mask] * y_gt[support_mask]
        self.X_svs = X[support_mask].copy()

        return None

        
    def fit(self: SVMC,
            X: np.ndarray,
            y_gt: np.ndarray,
            C: float = 1.0,
            epsilon: float = 1e-8, tol: float=1e-5,
            max_iters: int = int(1e6)) -> None:

        # make sure that the ground truth has elements {-1, +1}
        y_gt = self._to_expected_classes(y_gt)

        alphas, b = smo(self.kernel, X, y_gt, C=C, epsilon=epsilon, tol=tol, max_iters=max_iters)

        self._set_svs(alphas, X, y_gt, epsilon)
        self.b = b

        return None


    def svs_sum(self, X, K : Kernel):

        total = np.zeros(X.shape[0], dtype=float)
        for coefficient, support_vector in zip(self.sv_coeffs, self.X_svs):
            similarities = K.batch_similarity(X, support_vector).reshape(-1)
            total += coefficient * similarities

        return total


    def predict(self: SVMC,
                X: np.ndarray) -> np.ndarray:
        
        b = self.b
        K = self.kernel

        margins = self.svs_sum(X, K) + b
        y_hat = np.where(margins >= 0, 1, -1)

        return self._to_actual_classes(y_hat)
        

    def __str__(self: SVMC) -> str:
        return f"SVMC(header={self.header}, kernel={self.kernel}, nsvs={self.sv_coeffs.shape[0]}, b={self.b})"

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

        labels = y_gt.reshape(-1)
        classes = np.unique(labels)

        binary_header = Header()
        binary_header.features = list(self.header.features)
        binary_header.add_ground_truth(GroundTruth(("0", "1")))

        self.classes_ = classes
        self.constituent_svmcs = []
        for cls in classes:
            binary_labels = (labels == cls).astype(int)
            model = SVMC(binary_header, self.kernel)
            model.fit(X, binary_labels, C=C, epsilon=epsilon, max_iters=max_iters)
            self.constituent_svmcs.append(model)

    def predict(self: OVRC,
                X: np.ndarray) -> np.ndarray:

        scores = np.column_stack([
            model.b + model.svs_sum(X, model.kernel) for model in self.constituent_svmcs
        ])
        return self.classes_[np.argmax(scores, axis=1)]

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

        labels = y_gt.reshape(-1)
        classes = np.unique(labels)

        binary_header = Header()
        binary_header.features = list(self.header.features)
        binary_header.add_ground_truth(GroundTruth(("0", "1")))

        self.classes_ = classes
        self.constituent_svmcs = {}
        for i, cls1 in enumerate(classes):
            for cls2 in classes[i + 1:]:
                pair_mask = (labels == cls1) | (labels == cls2)
                binary_labels = (labels[pair_mask] == cls2).astype(int)
                model = SVMC(binary_header, self.kernel)
                model.fit(X[pair_mask], binary_labels,
                          C=C, epsilon=epsilon, max_iters=max_iters)
                self.constituent_svmcs[(cls1, cls2)] = model

    def predict(self: OVOC,
                X: np.ndarray) -> np.ndarray:

        votes = np.zeros((X.shape[0], len(self.classes_)), dtype=int)
        class_indices = {cls: i for i, cls in enumerate(self.classes_)}
        for (cls1, cls2), model in self.constituent_svmcs.items():
            predictions = model.predict(X)
            votes[:, class_indices[cls1]] += predictions == 0
            votes[:, class_indices[cls2]] += predictions == 1

        return self.classes_[np.argmax(votes, axis=1)]

    def __str__(self: OVOC) -> str:
        return f"OVOC(header={self.header}, kernel={self.kernel}, svmcs={self.svmcs})"

    def __repr__(self: OVOC) -> str:
        return str(self)

