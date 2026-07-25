"""Reference point noise (RPN) estimation and removal (Section III-B).

Informally the "Shenzhen algorithm": temporally deramp every point's time
series, then average the residuals across points per epoch to estimate the
reference point's own noise, and subtract it from every series.
"""
from __future__ import annotations

import numpy as np

from .library import design_matrix_h0


def temporal_deramp(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """OLS-remove a linear trend from a single point's time series y(t)."""
    A = design_matrix_h0(t)
    x_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ x_hat


def estimate_rpn(y_matrix: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Eq. (15): estimate the reference point error per epoch.

    y_matrix has shape (n_points, m_epochs). Returns a length-m vector
    e_hat(x_rp), the average deramped residual across all points per epoch.
    """
    deramped = np.array([temporal_deramp(y_matrix[i], t) for i in range(y_matrix.shape[0])])
    return deramped.mean(axis=0)


def remove_rpn(y_matrix: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eq. (16): subtract the estimated reference point error from every
    point's time series. Returns (y_prime_matrix, e_hat_rp).
    """
    e_hat_rp = estimate_rpn(y_matrix, t)
    return y_matrix - e_hat_rp, e_hat_rp
