import numpy as np

from insar_mht import hypotheses


def test_table_i_hypothesis_count_matches_formula():
    t = np.linspace(0, 5, 20)
    delta_temp = np.sin(2 * np.pi * t)
    beta_grid = [0.5, 1.0, 2.0]
    m = len(t)

    hyps = hypotheses.table_i_hypotheses(t, delta_temp, beta_grid)
    expected = 2 + 2 * len(beta_grid) + (m - 1) * (2 + len(beta_grid))
    assert len(hyps) == expected


def test_table_i_hypotheses_use_every_beta():
    t = np.linspace(0, 5, 20)
    delta_temp = np.sin(2 * np.pi * t)
    beta_grid = [0.5, 1.0, 2.0]

    hyps = hypotheses.table_i_hypotheses(t, delta_temp, beta_grid)
    names = [h.name for h in hyps]
    for beta in beta_grid:
        tag = f"beta={beta:g}"
        assert any(tag in n and n.startswith("H2_") for n in names)
        assert any(tag in n and n.startswith("H4_") for n in names)
        assert any(tag in n and n.startswith("H5_") for n in names)


def test_h4_columns_are_beta_specific():
    t = np.linspace(0, 5, 20)
    delta_temp = np.sin(2 * np.pi * t)
    beta_grid = [0.5, 5.0]

    hyps = {h.name: h for h in hypotheses.table_i_hypotheses(t, delta_temp, beta_grid)}
    h4_low = hyps["H4_exp+thermal(beta=0.5)"]
    h4_high = hyps["H4_exp+thermal(beta=5)"]
    assert not np.allclose(h4_low.C[:, 0], h4_high.C[:, 0])
