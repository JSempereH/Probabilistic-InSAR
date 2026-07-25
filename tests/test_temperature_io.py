import numpy as np
import pandas as pd
import pytest

from insar_mht.io import temperature


@pytest.fixture
def temp_csv(tmp_path):
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    p = tmp_path / "temp.csv"
    pd.DataFrame({"date": dates, "temp_mean_c": np.arange(10.0)}).to_csv(p, index=False)
    return p


def test_delta_temp_relative_to_first_date(temp_csv):
    temp = temperature.load_daily_temperature(temp_csv)
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-05", "2020-01-10"])
    delta = temperature.delta_temp_for_dates(temp, dates)
    assert np.allclose(delta, [0.0, 4.0, 9.0])


def test_delta_temp_raises_on_missing_date(temp_csv):
    temp = temperature.load_daily_temperature(temp_csv)
    dates = pd.DatetimeIndex(["2020-01-01", "2030-01-01"])
    with pytest.raises(ValueError):
        temperature.delta_temp_for_dates(temp, dates)
