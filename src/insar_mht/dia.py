"""Detection-Identification-Adaptation procedure (Section II-C, Fig. 2)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from . import testing as t
from .hypotheses import Hypothesis


@dataclass
class PrecomputedHypothesis:
    hypothesis: Hypothesis
    L: np.ndarray
    alpha_j: float
    critical_value: float


@dataclass
class DIAResult:
    rejected_h0: bool
    h0_fit: t.GLSFit
    best_hypothesis: Hypothesis | None
    best_ratio: float
    best_fit: t.GLSFit | None
    all_ratios: dict[str, float]


def precompute_hypothesis_class(
    hypothesis: Hypothesis, Qyy_inv: np.ndarray, q_ehat_ehat: np.ndarray, lambda0: float, gamma0: float
) -> PrecomputedHypothesis:
    """One-time setup per hypothesis *class* (Section III-D): L^j and the
    matched-power critical value depend only on C_j, Qyy and q_j, not on
    any individual point. Precompute these once, reuse for every point.
    """
    L = t.l_matrix(hypothesis.C, Qyy_inv, q_ehat_ehat)
    alpha_j = t.alpha_for_matched_power(lambda0, gamma0, hypothesis.q)
    critical_value = float(stats.chi2.ppf(1 - alpha_j, hypothesis.q))
    return PrecomputedHypothesis(hypothesis=hypothesis, L=L, alpha_j=alpha_j, critical_value=critical_value)


def run_dia(
    y: np.ndarray,
    A0: np.ndarray,
    Qyy: np.ndarray,
    precomputed: list[PrecomputedHypothesis],
    alpha0: float,
) -> DIAResult:
    """Per-point DIA loop (Fig. 2): detect via OMT, identify the best H_j by
    maximum test ratio, adapt by refitting the augmented model.
    """
    Qyy_inv = np.linalg.inv(Qyy)
    h0_fit = t.gls_fit(y, A0, Qyy_inv)

    q0 = A0.shape[0] - A0.shape[1]
    critical_value_omt = float(stats.chi2.ppf(1 - alpha0, q0))
    T0 = t.omt_statistic(h0_fit.e_hat, Qyy_inv)
    rejected = T0 > critical_value_omt

    if not rejected:
        return DIAResult(False, h0_fit, None, 0.0, None, {})

    ratios: dict[str, float] = {}
    best: PrecomputedHypothesis | None = None
    best_ratio = -np.inf
    for item in precomputed:
        Tq = t.test_statistic(h0_fit.e_hat, item.L)
        ratio = Tq / item.critical_value
        ratios[item.hypothesis.name] = ratio
        if ratio > best_ratio:
            best_ratio = ratio
            best = item

    if best is None or best_ratio <= 1.0:
        return DIAResult(True, h0_fit, None, best_ratio, None, ratios)

    A_augmented = np.column_stack([A0, best.hypothesis.C])
    best_fit = t.gls_fit(y, A_augmented, Qyy_inv)
    return DIAResult(True, h0_fit, best.hypothesis, best_ratio, best_fit, ratios)
