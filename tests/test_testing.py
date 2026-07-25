import numpy as np
import pytest

from insar_mht import testing as t
from insar_mht.library import design_matrix_h0, m6_heaviside_step


def test_gls_fit_recovers_noise_free_linear_trend():
    time = np.arange(20.0)
    y = 3.0 * time
    A = design_matrix_h0(time)
    Qyy_inv = np.eye(len(time))
    fit = t.gls_fit(y, A, Qyy_inv)
    assert np.allclose(fit.x_hat, [3.0])
    assert np.allclose(fit.e_hat, 0.0, atol=1e-10)


def test_omt_statistic_is_zero_for_perfect_fit():
    time = np.arange(20.0)
    y = 3.0 * time
    A = design_matrix_h0(time)
    Qyy_inv = np.eye(len(time))
    fit = t.gls_fit(y, A, Qyy_inv)
    assert t.omt_statistic(fit.e_hat, Qyy_inv) == pytest.approx(0.0, abs=1e-10)


def test_l_matrix_test_statistic_matches_direct_computation():
    time = np.arange(20.0)
    rng = np.random.default_rng(0)
    y = 2.0 * time + rng.normal(scale=0.1, size=time.shape)
    A0 = design_matrix_h0(time)
    Qyy = np.eye(len(time))
    Qyy_inv = np.linalg.inv(Qyy)

    h0_fit = t.gls_fit(y, A0, Qyy_inv)
    C = m6_heaviside_step(len(time), 10).reshape(-1, 1)

    L = t.l_matrix(C, Qyy_inv, h0_fit.q_ehat_ehat)
    Tq_efficient = t.test_statistic(h0_fit.e_hat, L)

    # direct computation of Eq. (5) without the trace trick
    middle = C.T @ Qyy_inv @ h0_fit.q_ehat_ehat @ Qyy_inv @ C
    Tq_direct = float(h0_fit.e_hat.T @ Qyy_inv @ C @ np.linalg.inv(middle) @ C.T @ Qyy_inv @ h0_fit.e_hat)

    assert Tq_efficient == pytest.approx(Tq_direct)


def test_step_detected_when_present_in_data():
    time = np.arange(30.0)
    y = 1.0 * time
    y[15:] += 20.0  # large offset, should dominate over linear-only H0
    A0 = design_matrix_h0(time)
    Qyy = np.eye(len(time))
    Qyy_inv = np.linalg.inv(Qyy)

    h0_fit = t.gls_fit(y, A0, Qyy_inv)
    C = m6_heaviside_step(len(time), 15).reshape(-1, 1)
    L = t.l_matrix(C, Qyy_inv, h0_fit.q_ehat_ehat)
    Tq = t.test_statistic(h0_fit.e_hat, L)

    alpha = 0.05
    ratio = t.test_ratio(Tq, q=1, alpha=alpha)
    assert ratio > 1.0


def test_noncentrality_from_power_increases_with_gamma():
    lam_low = t.noncentrality_from_power(gamma0=0.3, alpha=0.01, q=1)
    lam_high = t.noncentrality_from_power(gamma0=0.8, alpha=0.01, q=1)
    assert lam_high > lam_low


def test_alpha_for_matched_power_reproduces_reference_lambda():
    alpha0, gamma0, q = 0.01, 0.5, 1
    lambda0 = t.noncentrality_from_power(gamma0, alpha0, q)
    alpha_roundtrip = t.alpha_for_matched_power(lambda0, gamma0, q)
    assert alpha_roundtrip == pytest.approx(alpha0, rel=1e-4)
