import zipfile

import numpy as np
import pytest

from insar_mht.io import egms

HEADER = (
    "pid,mp_type,latitude,longitude,easting,northing,height_ortho,height_ellipse,"
    "line,pixel,rmse_ts,temporal_coherence,amplitude_dispersion,incidence_angle,"
    "track_angle,los_east,los_north,los_up,mean_velocity,mean_velocity_std,"
    "acceleration,acceleration_std,seasonality,seasonality_std,gnss_velocity,"
    "20200103,20200109,20200115\n"
)
ROWS = [
    "A1,0,38.16,-0.47,3399797,1738758,1.4,51.6,528,957,2,0.9,0.35,30.97,347.64,"
    "-0.511,-0.112,0.852,0.2,0.1,-0.2,0.16,0.5,0.1,-0.4,0.0,1.2,2.4\n",
    "B2,0,38.20,-0.40,3400000,1739000,1.0,50.0,500,900,2,0.9,0.35,30.97,347.64,"
    "-0.511,-0.112,0.852,0.1,0.1,-0.1,0.16,0.4,0.1,-0.3,1.0,1.5,1.8\n",
]


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "sample.csv"
    p.write_text(HEADER + "".join(ROWS))
    return p


@pytest.fixture
def zip_path(tmp_path, csv_path):
    p = tmp_path / "sample.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.write(csv_path, arcname="sample.csv")
        z.writestr("sample.xml", "<metadata/>")
    return p


def test_load_egms_csv_parses_bare_date_columns(csv_path):
    ds = egms.load_egms_csv(csv_path)
    assert len(ds.dates) == 3
    assert list(ds.metadata["pid"]) == ["A1", "B2"]
    assert set(ds.displacements.columns) == {"pid", "date", "displacement_mm"}


def test_load_egms_csv_from_zip_matches_plain_csv(csv_path, zip_path):
    ds_csv = egms.load_egms_csv(csv_path)
    ds_zip = egms.load_egms_csv(zip_path)
    assert ds_csv.metadata.equals(ds_zip.metadata)
    assert ds_csv.displacements.equals(ds_zip.displacements)


def test_point_time_series_starts_at_zero(csv_path):
    ds = egms.load_egms_csv(csv_path)
    t, y = egms.point_time_series(ds, "A1")
    assert t[0] == 0.0
    assert np.allclose(y, [0.0, 1.2, 2.4])
    assert len(t) == len(y) == 3


def test_load_egms_bbox_filters_points(zip_path):
    ds = egms.load_egms_bbox(zip_path, min_lon=-0.5, min_lat=38.1, max_lon=-0.45, max_lat=38.19, chunksize=1)
    assert list(ds.metadata["pid"]) == ["A1"]


def test_load_many_bbox_skips_files_with_no_coverage(zip_path, tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text(HEADER + ROWS[1])  # only B2, which is outside the bbox below
    ds = egms.load_many_bbox([zip_path, empty_csv], min_lon=-0.5, min_lat=38.1, max_lon=-0.45, max_lat=38.19)
    assert list(ds.metadata["pid"]) == ["A1"]


def test_load_egms_bbox_raises_when_empty(zip_path):
    with pytest.raises(ValueError):
        egms.load_egms_bbox(zip_path, min_lon=10, min_lat=10, max_lon=11, max_lat=11)
