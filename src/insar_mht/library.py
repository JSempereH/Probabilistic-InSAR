"""Canonical kinematic functions M1-M6 from Chang & Hanssen (2016), Eq. (14).

Each function returns a single column (shape (m,)) to be placed in a design
or specification matrix. The physical parameter each column multiplies
(v, eta, s, c, kappa/beta, D_i, Delta_i) is estimated separately by GLS;
these functions only build the *known* part of the model.
"""
from __future__ import annotations

import numpy as np


def m1_linear(t: np.ndarray) -> np.ndarray:
    """Steady-state (linear) motion column: t * v."""
    return np.asarray(t, dtype=float)


def m2_thermal_expansion(delta_temp: np.ndarray) -> np.ndarray:
    """Thermal expansion column: delta_T * eta."""
    return np.asarray(delta_temp, dtype=float)


def m3_seasonal(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Seasonal (1-year period) columns for amplitude terms s and c.

    Returns (sin_col, cos_col) such that the model is
    sin(2*pi*t) * s + (cos(2*pi*t) - 1) * c, with t in years.
    """
    t = np.asarray(t, dtype=float)
    return np.sin(2 * np.pi * t), np.cos(2 * np.pi * t) - 1.0


def m4_exponential_relaxation(t: np.ndarray, beta: float) -> np.ndarray:
    """Exponential relaxation column for a *fixed* decay constant beta.

    Nonlinear in beta: within MHT screening, beta is treated as fixed
    (e.g. tested over a grid of candidate values, each its own hypothesis).
    Once the optimal model is identified, refine beta by Taylor-linearized
    least squares (see Section II-D of the paper); not implemented here.

    Keep candidate beta values well below the total observation span. As
    beta approaches the span, 1 - exp(-t/beta) becomes nearly linear in t
    and this column gets collinear with the null model's linear trend
    (M1), which makes the individual velocity and kappa estimates unstable
    even though the combined fit can look fine.
    """
    t = np.asarray(t, dtype=float)
    return 1.0 - np.exp(-t / beta)


def m5_outlier(m: int, epoch_index: int) -> np.ndarray:
    """Kronecker-delta column: a single-epoch outlier D_i at epoch_index."""
    col = np.zeros(m)
    col[epoch_index] = 1.0
    return col


def m6_heaviside_step(m: int, step_index: int) -> np.ndarray:
    """Step column: 0 before step_index, 1 from step_index onward.

    step_index is the first (0-indexed) epoch affected by the offset,
    i.e. the step occurs between epoch step_index - 1 and step_index.
    """
    col = np.zeros(m)
    col[step_index:] = 1.0
    return col


def design_matrix_h0(t: np.ndarray) -> np.ndarray:
    """Null hypothesis design matrix A: a single column of temporal baselines."""
    return np.asarray(t, dtype=float).reshape(-1, 1)
