"""Library of alternative hypotheses H_j, built from the canonical functions.

Reproduces Table I (H1-H7) plus the per-epoch M5 (outlier) and M6 (step)
families used for unwrapping-error detection and generic step search.
Every hypothesis involving M4 (exponential relaxation, nonlinear in beta)
is built once per candidate value in beta_grid, since beta cannot be
estimated directly by the screening test (Section II-D).

The paper notes the full library can include "many (nested) combinations"
of the canonical functions beyond Table I's seven; this module ships
exactly those seven, which is what the rest of the pipeline expects.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import library as lib


@dataclass
class Hypothesis:
    name: str
    C: np.ndarray  # m x q specification matrix
    q: int


def table_i_hypotheses(t: np.ndarray, delta_temp: np.ndarray, beta_grid: list[float]) -> list[Hypothesis]:
    """Build H1-H7 from Table I for a single point's time vector t (years)
    and temperature differences delta_temp (same length as t).

    H1: M1 + M2                (q=1)
    H2: M4(exp) + M4 slope     (exponential replaces the linear model, one per beta_grid)
    H3: M1 + M3 (seasonal)     (q=2)
    H4: M4 + M2                (q=2, one per beta_grid)
    H5: M4 + M6                (q=2, one per beta_grid x step_index)
    H6: M1 + M2 + M6           (q=2)
    H7: M1 + M2 + M3 + M6      (q=4)

    Offsets (M6) in H5-H7 are left as free per-epoch searches: build one
    Hypothesis per candidate step_index and let the DIA loop pick the best.
    Total hypothesis count is 2 + 2*len(beta_grid) + (m-1) * (2 + len(beta_grid)).
    """
    t = np.asarray(t, dtype=float)
    m = len(t)
    m2 = lib.m2_thermal_expansion(delta_temp)
    m3_sin, m3_cos = lib.m3_seasonal(t)
    m4_by_beta = [(beta, lib.m4_exponential_relaxation(t, beta)) for beta in beta_grid]

    # H1's C matrix only carries M2: M1 (v * t) is already in the null
    # design matrix A, so it is not part of the additional-parameter test.
    # H2 is the odd one out per the Table I caption: "when an exponential
    # function is tested, it replaces the linear model M1". So its full
    # model (for the later Adaptation/estimation step, not shown here) uses
    # A_H2 = [m4] instead of A_H0 = [t]. The C matrix below is still correct
    # for the *screening* test since it is compared against the same H0.
    hypotheses = []
    hypotheses.append(Hypothesis("H1_linear+thermal", np.column_stack([m2]), 1))
    hypotheses.append(Hypothesis("H3_seasonal", np.column_stack([m3_sin, m3_cos]), 2))

    for beta, m4 in m4_by_beta:
        hypotheses.append(Hypothesis(f"H2_exp(beta={beta:g})", np.column_stack([m4]), 1))
        hypotheses.append(Hypothesis(f"H4_exp+thermal(beta={beta:g})", np.column_stack([m4, m2]), 2))

    for step_index in range(1, m):
        m6 = lib.m6_heaviside_step(m, step_index)
        for beta, m4 in m4_by_beta:
            hypotheses.append(Hypothesis(f"H5_exp+step@{step_index}(beta={beta:g})", np.column_stack([m4, m6]), 2))
        hypotheses.append(Hypothesis(f"H6_thermal+step@{step_index}", np.column_stack([m2, m6]), 2))
        hypotheses.append(
            Hypothesis(f"H7_thermal+seasonal+step@{step_index}", np.column_stack([m2, m3_sin, m3_cos, m6]), 4)
        )
    return hypotheses


def outlier_hypotheses(m: int) -> list[Hypothesis]:
    """M5 family: one single-epoch outlier hypothesis per epoch (q=1 each)."""
    return [Hypothesis(f"M5_outlier@{i}", lib.m5_outlier(m, i).reshape(-1, 1), 1) for i in range(m)]


def step_hypotheses(m: int) -> list[Hypothesis]:
    """M6 family: one discrete-step hypothesis per epoch transition (q=1 each)."""
    return [Hypothesis(f"M6_step@{i}", lib.m6_heaviside_step(m, i).reshape(-1, 1), 1) for i in range(1, m)]
