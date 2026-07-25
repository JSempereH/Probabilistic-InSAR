"""Precision and reliability evaluation of an estimated model (Section II-C.3).

Qxx (Eq. 11) is the parameter covariance under the identified optimal model.
Comparing it against a criterion matrix H_x via the generalized eigenvalue
problem (Eq. 13) tells us whether the achieved precision meets a predefined
target: eigenvalues <= 1 mean the estimate is at least as precise as required.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def q_xx(A_augmented: np.ndarray, Qyy_inv: np.ndarray) -> np.ndarray:
    """Eq. (11): Qxx = ([A : Cj]' Qyy^-1 [A : Cj])^-1."""
    normal_matrix = A_augmented.T @ Qyy_inv @ A_augmented
    return np.linalg.inv(normal_matrix)


def precision_eigenvalues(Qxx: np.ndarray, Hx: np.ndarray) -> np.ndarray:
    """Eq. (13): generalized eigenvalues of |Qxx - zeta * Hx| = 0.

    zeta == 1 for all parameters means Qxx matches the criterion exactly;
    zeta < 1 is better than required, zeta > 1 is worse. max(zeta) is a
    single global precision metric (zeta_max in the paper).
    """
    eigenvalues = eigh(Qxx, Hx, eigvals_only=True)
    return np.sort(eigenvalues)
