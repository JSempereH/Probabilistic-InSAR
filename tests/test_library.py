import numpy as np

from insar_mht import library as lib


def test_m1_linear_is_identity_on_t():
    t = np.array([0.0, 1.0, 2.5])
    assert np.allclose(lib.m1_linear(t), t)


def test_m5_outlier_is_unit_vector():
    col = lib.m5_outlier(5, 2)
    assert col.sum() == 1.0
    assert col[2] == 1.0


def test_m6_heaviside_step_shape():
    col = lib.m6_heaviside_step(6, 3)
    assert np.allclose(col, [0, 0, 0, 1, 1, 1])


def test_m3_seasonal_period_is_one_year():
    t = np.array([0.0, 1.0, 2.0])
    sin_col, cos_col = lib.m3_seasonal(t)
    assert np.allclose(sin_col, 0.0, atol=1e-10)
    assert np.allclose(cos_col, 0.0, atol=1e-10)


def test_design_matrix_h0_shape():
    t = np.arange(10.0)
    A = lib.design_matrix_h0(t)
    assert A.shape == (10, 1)
