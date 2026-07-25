"""Core numerics for MHT / B-method of testing, Section II of Chang & Hanssen (2016).

Naming follows the paper: y (observations), A (H0 design matrix), Qyy
(covariance), C_j (specification matrix of an alternative hypothesis),
q_j (its degrees of freedom), L^j (its precomputed test matrix).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import optimize, stats


@dataclass
class GLSFit:
    x_hat: np.ndarray
    e_hat: np.ndarray
    y_hat: np.ndarray
    q_ehat_ehat: np.ndarray


def gls_fit(y: np.ndarray, A: np.ndarray, Qyy_inv: np.ndarray) -> GLSFit:
    """Generalized least squares fit under a hypothesis with design matrix A.

    Eq. (4): Q_ehat_ehat = Qyy - A (A' Qyy^-1 A)^-1 A'
    """
    normal_matrix = A.T @ Qyy_inv @ A
    n_inv = np.linalg.inv(normal_matrix)
    x_hat = n_inv @ A.T @ Qyy_inv @ y
    y_hat = A @ x_hat
    e_hat = y - y_hat
    qyy = np.linalg.inv(Qyy_inv)
    q_ehat_ehat = qyy - A @ n_inv @ A.T
    return GLSFit(x_hat=x_hat, e_hat=e_hat, y_hat=y_hat, q_ehat_ehat=q_ehat_ehat)


def omt_statistic(e_hat: np.ndarray, Qyy_inv: np.ndarray) -> float:
    """Overall model test statistic, Eq. (10): T0 = e_hat' Qyy^-1 e_hat."""
    return float(e_hat.T @ Qyy_inv @ e_hat)


def l_matrix(C: np.ndarray, Qyy_inv: np.ndarray, q_ehat_ehat: np.ndarray) -> np.ndarray:
    """Precomputed test matrix for a hypothesis class, Eq. (5).

    L^j = Qyy^-1 C (C' Qyy^-1 Q_ehat0ehat0 Qyy^-1 C)^-1 C' Qyy^-1

    Depends only on C, Qyy and the null-hypothesis residual covariance --
    not on any individual point's data. Compute once per hypothesis class.
    """
    middle = C.T @ Qyy_inv @ q_ehat_ehat @ Qyy_inv @ C
    middle_inv = np.linalg.inv(np.atleast_2d(middle))
    return Qyy_inv @ C @ middle_inv @ C.T @ Qyy_inv


def test_statistic(e_hat0: np.ndarray, L: np.ndarray) -> float:
    """Efficient test statistic, Eq. (17): T_q^j = e_hat0' L^j e_hat0.

    Equivalent to tr(L (e_hat0 e_hat0')): the trace form is what makes
    per-point evaluation cheap once e_hat0 e_hat0' is computed (see
    notebook 05 / blog post IV for the vectorized version).
    """
    return float(e_hat0.T @ L @ e_hat0)


def noncentrality_from_power(gamma0: float, alpha: float, q: int) -> float:
    """Invert Eq. (8): given power gamma0, significance alpha and dof q,
    find the noncentrality parameter lambda0 such that

        gamma0 = 1 - ncx2.cdf(chi2.ppf(1 - alpha, q), q, lambda0)
    """
    critical_value = float(stats.chi2.ppf(1 - alpha, q))

    def power_gap(lam: float) -> float:
        power = 1.0 - stats.ncx2.cdf(critical_value, q, lam)
        return float(power - gamma0)

    return float(cast(float, optimize.brentq(power_gap, 1e-8, 1e4)))


def alpha_for_matched_power(lambda0: float, gamma0: float, q: int) -> float:
    """B-method calibration: given a fixed reference noncentrality lambda0
    (derived once from alpha0, gamma0, q=1) and a new dimension q, find the
    significance level alpha_j that keeps the power at gamma0 (Section II-B).
    """

    def power_gap(alpha: float) -> float:
        critical_value = stats.chi2.ppf(1 - alpha, q)
        power = 1.0 - stats.ncx2.cdf(critical_value, q, lambda0)
        return float(power - gamma0)

    return float(cast(float, optimize.brentq(power_gap, 1e-12, 1 - 1e-12)))


def test_ratio(Tq: float, q: int, alpha: float) -> float:
    """Eq. (6): normalize a test statistic by its critical value so that
    hypotheses of different dimension become comparable.
    """
    return float(Tq / stats.chi2.ppf(1 - alpha, q))
