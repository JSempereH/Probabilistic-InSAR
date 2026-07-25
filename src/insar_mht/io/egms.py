"""Reader for EGMS (European Ground Motion Service) CSV exports.

EGMS point products (Basic L2a / Calibrated L2b / Ortho L3) are distributed
as track/burst-level CSV files (one row per point, packaged as .zip with a
sibling .xml metadata file) with columns:

    pid, mp_type, latitude, longitude, easting, northing,
    height_ortho, height_ellipse, line, pixel,
    rmse_ts, temporal_coherence, amplitude_dispersion,
    incidence_angle, track_angle, los_east, los_north, los_up,
    mean_velocity, mean_velocity_std, acceleration, acceleration_std,
    seasonality, seasonality_std, gnss_velocity,
    20200103, 20200109, ...   (one column per acquisition date, bare
                                YYYYMMDD with no prefix, cumulative LOS
                                displacement in mm relative to the first
                                epoch and the product's spatial reference)

These files are large (a single Sentinel-1 track/burst easily reaches
hundreds of MB to ~1 GB uncompressed) and normally cover a much bigger
area than a single AOI, so `load_egms_bbox` streams the file in chunks
and keeps only points inside a given lon/lat box. The whole-file
`load_egms_csv` is only practical once you already have a small,
pre-clipped export.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

_DATE_COL_RE = re.compile(r"^(\d{8})$")


@dataclass
class EGMSDataset:
    metadata: pd.DataFrame  # one row per point: pid, easting, northing, mean_velocity, ...
    displacements: pd.DataFrame  # long format: pid, date, displacement_mm
    dates: pd.DatetimeIndex


def _open_csv_member(path: str | Path) -> IO[bytes]:
    """Return a readable binary stream for the CSV, transparently unzipping
    if `path` is one of the EGMS .zip downloads (csv + xml pair).
    """
    path = Path(path)
    if path.suffix == ".zip":
        z = zipfile.ZipFile(path)
        csv_name = next(n for n in z.namelist() if n.endswith(".csv"))
        return z.open(csv_name)
    return open(path, "rb")


def _dates_from_columns(date_cols: list[str]) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(pd.to_datetime(date_cols, format="%Y%m%d")))


def _to_long_displacements(raw: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    long = raw.melt(id_vars=["pid"], value_vars=date_cols, var_name="date_col", value_name="displacement_mm")
    long["date"] = pd.to_datetime(long["date_col"], format="%Y%m%d")
    return long.drop(columns="date_col").sort_values(["pid", "date"]).reset_index(drop=True)


def load_egms_csv(path: str | Path) -> EGMSDataset:
    """Load an EGMS export (.csv or .zip) fully into memory.

    Only use this on files you already know are small (e.g. an export
    that was clipped to your AOI at download time). For the raw
    track/burst-level files from the bulk API, use `load_egms_bbox`.
    """
    raw = pd.read_csv(_open_csv_member(path))
    date_cols = [c for c in raw.columns if _DATE_COL_RE.match(c)]
    metadata = raw[[c for c in raw.columns if c not in date_cols]].copy()
    displacements = _to_long_displacements(raw, date_cols)
    return EGMSDataset(metadata=metadata, displacements=displacements, dates=_dates_from_columns(date_cols))


def load_cached(metadata_path: str | Path, displacements_path: str | Path) -> EGMSDataset:
    """Load an AOI previously cached to parquet by `EGMSDataset` (see
    notebook 00's caching cell), skipping the expensive zip re-scan.
    """
    metadata = pd.read_parquet(metadata_path)
    displacements = pd.read_parquet(displacements_path)
    dates = pd.DatetimeIndex(sorted(displacements["date"].unique()))
    return EGMSDataset(metadata=metadata, displacements=displacements, dates=dates)


def load_egms_bbox(
    path: str | Path,
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    chunksize: int = 200_000,
) -> EGMSDataset:
    """Stream a large EGMS track/burst export (.csv or .zip) and keep only
    the points that fall inside the given lon/lat bounding box.

    This is the entry point to use for the raw files the bulk-download API
    hands you. It never holds the full file in memory.
    """
    date_cols: list[str] | None = None
    kept_chunks: list[pd.DataFrame] = []

    reader = pd.read_csv(_open_csv_member(path), chunksize=chunksize)
    for chunk in reader:
        if date_cols is None:
            date_cols = [c for c in chunk.columns if _DATE_COL_RE.match(c)]
        mask = chunk["longitude"].between(min_lon, max_lon) & chunk["latitude"].between(min_lat, max_lat)
        if mask.any():
            kept_chunks.append(chunk.loc[mask])

    if date_cols is None or not kept_chunks:
        raise ValueError(f"No points found in bbox ({min_lon},{min_lat},{max_lon},{max_lat}) for {path}")

    raw = pd.concat(kept_chunks, ignore_index=True)
    metadata = raw[[c for c in raw.columns if c not in date_cols]].copy()
    displacements = _to_long_displacements(raw, date_cols)
    return EGMSDataset(metadata=metadata, displacements=displacements, dates=_dates_from_columns(date_cols))


def load_many_bbox(paths: list[str | Path], min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> EGMSDataset:
    """Load several track/burst files (e.g. overlapping tracks over the
    same AOI) and concatenate them into one dataset. Points from different
    tracks have different LOS geometries (see los_east/los_north/los_up in
    the metadata); do not average displacements across tracks for the
    same physical location without accounting for that.
    """
    datasets = []
    for p in paths:
        try:
            datasets.append(load_egms_bbox(p, min_lon, min_lat, max_lon, max_lat))
        except ValueError:
            # Normal case: bursts/tracks tile an area, so a given AOI often
            # only intersects a subset of the files passed in.
            continue

    if not datasets:
        raise ValueError(f"No points found in bbox ({min_lon},{min_lat},{max_lon},{max_lat}) in any of {paths}")

    metadata = pd.concat([d.metadata for d in datasets], ignore_index=True)
    displacements = pd.concat([d.displacements for d in datasets], ignore_index=True)
    dates = datasets[0].dates
    return EGMSDataset(metadata=metadata, displacements=displacements, dates=dates)


def point_time_series(dataset: EGMSDataset, pid) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_years, y_mm) for a single point, t measured from its first
    acquisition. Ready to feed into insar_mht.library.design_matrix_h0.
    """
    _, t_years, y_mm = point_series(dataset, pid)
    return t_years, y_mm


def point_series(dataset: EGMSDataset, pid) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Return (dates, t_years, y_mm) for a single point. Use this instead of
    `point_time_series` when you also need the acquisition dates, e.g. to
    align a temperature record for the M2 thermal-expansion column.
    """
    series = dataset.displacements.loc[dataset.displacements["pid"] == pid].sort_values("date")
    dates = pd.DatetimeIndex(series["date"])
    t_years = (dates - dates[0]).days.to_numpy() / 365.25
    y_mm = series["displacement_mm"].to_numpy()
    return dates, t_years, y_mm


def points_in_bbox(dataset: EGMSDataset, min_lon: float, min_lat: float, max_lon: float, max_lat: float) -> pd.DataFrame:
    """Filter an already-loaded dataset's metadata to a lon/lat bounding box."""
    m = dataset.metadata
    mask = m["longitude"].between(min_lon, max_lon) & m["latitude"].between(min_lat, max_lat)
    return m.loc[mask]
