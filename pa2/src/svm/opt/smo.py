# SYSTEM IMPORTS
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Sequence, Set
from typing import Tuple, Union
import numpy as np


# PYTHON PROJECT IMPORTS
from ..kernels.kernel import Kernel

def model_output(K: Kernel, alphas, X, y_gt, b, k):
    alphas = alphas.reshape(-1)
    y_gt = y_gt.reshape(-1)
    result = 0
    for i in range(len(alphas)):
        result += alphas[i]*y_gt[i]*K.sample_similarity(X[i], X[k])
    return result + b

def _get_errors(K: Kernel,
                alphas: np.ndarray, b: float,
                X: np.ndarray, y_gt: np.ndarray) -> np.ndarray:

    y_gt = y_gt.reshape(-1)
    errors = [model_output(K, alphas, X, y_gt, b, i) - y_gt[i] for i in range(len(y_gt))]

    return np.asarray(errors)


def _alphas_violating_kkt(K: Kernel,
                          alphas: np.ndarray, b: float,
                          errors: np.ndarray,
                          X: np.ndarray, y_gt: np.ndarray, C: float,
                          epsilon, tol: float) -> Sequence[int]:
    
    alphas = alphas.reshape(-1)
    signed_errors = y_gt.reshape(-1) * np.asarray(errors).reshape(-1)
    violations = []

    for i in range(len(signed_errors)):
        if ((alphas[i] < C - epsilon and signed_errors[i] < -tol) or (alphas[i] > epsilon and signed_errors[i] > tol)):
            violations.append(i)

    return violations


def _choose_alpha_violating_kkt(K: Kernel,
                                alphas: np.ndarray, b: float,
                                errors: np.ndarray,
                                X: np.ndarray, y_gt: np.ndarray, C: float,
                                epsilon, tol: float) -> int:

    violations = _alphas_violating_kkt(
        K, alphas, b, errors, X, y_gt, C, epsilon, tol
    )
    return int(np.random.choice(violations))


def _smo_step(K: Kernel,
              alphas: np.ndarray, b: float,
              i1: int, i2: int,
              errors: np.ndarray,
              X: np.ndarray, y_gt: np.ndarray, C: float,
              epsilon: float, tol: float) -> tuple[bool, np.ndarray, float]:

    if i1 == i2:
        return False, alphas, b

    y_gt = y_gt.reshape(-1)
    errors = np.asarray(errors).reshape(-1)
    a1, a2 = float(alphas.flat[i1]), float(alphas.flat[i2])
    y1, y2 = y_gt[i1], y_gt[i2]
    e1, e2 = errors[i1], errors[i2]

    if y1 == y2: 
        L = max(0, a1+a2-C)
        H = min(C, a1+a2)
    else:
        L = max(0, a2-a1)
        H = min(C, C+a2-a1)

    if L >= H: 
        return False, alphas, b


    k11 = K.sample_similarity(X[i1], X[i1])
    k12 = K.sample_similarity(X[i1], X[i2])
    k21 = K.sample_similarity(X[i2], X[i1])
    k22 = K.sample_similarity(X[i2], X[i2])

    eta = 2 * k21 - k11 - k22
    if eta >= 0: 
        return False, alphas, b

    a2_new = a2+(y2*(e2-e1)/eta)
    if a2_new > H:
        a2_new = H
    elif a2_new < L:
        a2_new = L

    if a2_new == a2 or abs(a2_new - a2) < epsilon:
        return False, alphas, b


    a1_new = a1-y1*y2*(a2_new-a2)

    b1 = b - e1 - y1*(a1_new-a1)*k11-y2*(a2_new-a2)*k21
    b2 = b - e2 - y1*(a1_new-a1)*k12-y2*(a2_new-a2)*k22

    if a1_new > 0 and a1_new < C:
        b = b1
    elif a2_new > 0 and a2_new < C:
        b = b2
    else:
        b = (b1+b2)/2

    alphas.flat[i1] = a1_new
    alphas.flat[i2] = a2_new
    return True, alphas, float(b)


def smo(K: Kernel,
        X: np.ndarray, y_gt: np.ndarray, C: float = 1.0,
        epsilon: float = 1e-8,
        tol: float = 1e-5,
        max_iters: int = int(1e6)) -> tuple[np.ndarray, float]:

    num_samples = len(y_gt)
    alphas = np.zeros(num_samples)
    b = 0
    it = 0

    while it < max_iters:
        it += 1
        errors = _get_errors(K, alphas, b, X, y_gt)
        violations = _alphas_violating_kkt(K, alphas, b, errors, X, y_gt, C, epsilon, tol)
        if not violations:
            break
        i1 = _choose_alpha_violating_kkt(K, alphas, b, errors, X, y_gt, C, epsilon, tol)
        for i2 in range(num_samples):
            if i2 == i1:
                continue
            changed, alphas, b = _smo_step(K, alphas, b, i1, i2, errors, X, y_gt, C, epsilon, tol)
            if changed:
                break
        
    return alphas, b

