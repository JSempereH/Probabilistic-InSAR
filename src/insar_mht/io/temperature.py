"""Daily mean temperature record, for the M2 thermal-expansion column (Eq. 14).

The paper uses the actual temperature record for the acquisition dates
(Fig. 3a: "temperature differences between the acquired dates of the
master image and the slaves"). `data/processed/alicante_daily_temperature.csv`
was pulled from Open-Meteo's ERA5-based historical archive
(https://archive-api.open-meteo.com/v1/archive), which needs no API key
and covers any lon/lat back to 1940: a reasonable free stand-in for a
real station/reanalysis record.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_daily_temperature(path: str | Path) -> pd.Series:
    """Return a Series of daily mean temperature (deg C) indexed by date."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date")["temp_mean_c"]


def delta_temp_for_dates(temperature: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Eq. 14's delta_T: temperature at each acquisition date minus the
    temperature at the first (reference/master) date.
    """
    values = temperature.reindex(dates)
    if values.isna().any():
        missing = dates[values.isna()]
        raise ValueError(f"no temperature record for dates: {list(missing)}")
    return (values - values.iloc[0]).to_numpy()
