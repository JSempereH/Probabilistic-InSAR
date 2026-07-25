"""Residual phase-unwrapping error detection and correction (Section III-C).

Strategy: run DIA restricted to the M5 (single-epoch outlier) and M6
(discrete step) hypothesis families. If the winning hypothesis's estimated
offset magnitude exceeds wavelength / 4, it is treated as an unwrapping
error (a true deformation offset is expected to be much smaller than a
half-wavelength) and the observations from that epoch onward are corrected
by minus the nearest half-wavelength step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import dia as dia_mod
from .hypotheses import outlier_hypotheses, step_hypotheses


@dataclass
class UnwrapCorrection:
    epoch_index: int
    hypothesis_name: str
    estimated_offset: float
    correction: float


def detect_and_correct(
    y: np.ndarray,
    A0: np.ndarray,
    Qyy: np.ndarray,
    t: np.ndarray,
    wavelength: float,
    alpha0: float,
    gamma0: float,
) -> tuple[np.ndarray, list[UnwrapCorrection]]:
    """Single-pass detection + correction. For repeated cycle slips in the
    same series, call this in a loop until it returns no corrections
    (see notebook 06).
    """
    from . import testing as tst

    m = len(t)
    Qyy_inv = np.linalg.inv(Qyy)
    h0_fit = tst.gls_fit(y, A0, Qyy_inv)
    lambda0 = tst.noncentrality_from_power(gamma0, alpha0, q=1)

    candidates = outlier_hypotheses(m) + step_hypotheses(m)
    precomputed = [
        dia_mod.precompute_hypothesis_class(h, Qyy_inv, h0_fit.q_ehat_ehat, lambda0, gamma0) for h in candidates
    ]

    result = dia_mod.run_dia(y, A0, Qyy, precomputed, alpha0)
    if not result.rejected_h0 or result.best_hypothesis is None or result.best_fit is None:
        return y, []

    offset_estimate = float(result.best_fit.x_hat[-1])
    if abs(offset_estimate) <= wavelength / 4:
        return y, []

    correction = -np.sign(offset_estimate) * wavelength / 2
    is_step = result.best_hypothesis.name.startswith("M6_step")
    epoch_index = int(result.best_hypothesis.name.rsplit("@", 1)[-1])

    y_corrected = y.copy()
    if is_step:
        y_corrected[epoch_index:] += correction
    else:
        y_corrected[epoch_index] += correction

    return y_corrected, [UnwrapCorrection(epoch_index, result.best_hypothesis.name, offset_estimate, correction)]
