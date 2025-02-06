import logging

import geotool
import numpy as np
import pytest

log = logging.getLogger(__name__)


def test_lbh2xyz():
    x, y, z = geotool.lbh2xyz(48.8566, 2.3522, 35.0)
    assert x == pytest.approx(4192979.6198897623)
    assert y == pytest.approx(4799159.563725418)
    assert z == pytest.approx(260022.66015989496)


def test_xyz2lbh():
    lon, lat, h = geotool.xyz2lbh(4192979.6198897623, 4799159.563725418, 260022.66015989496)
    assert lon == pytest.approx(48.8566)
    assert lat == pytest.approx(2.3522)
    assert h == pytest.approx(35, abs=1e-3)


# region CoordMigrate


# endregion
# region datum_compense
def test_datum_compense_scalar():
    expected_x, expected_y = 469706.56912942487, 2821763.831232311
    test_x, test_y = geotool.datum_compense(xc=469704.6693, yc=2821940.796, hb=400)

    assert test_x == pytest.approx(expected_x)
    assert test_y == pytest.approx(expected_y)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        pytest.param(
            np.array([469704.6693, 469704.6693], np.float64),
            np.array([2821940.796, 2821940.796], np.float64),
            id="ndarry-ndarry",
        ),
    ],
)
def test_datum_compense_vector(x, y):
    expected_x = (469706.56912942487, 469706.56912942487)
    expected_y = (2821763.831232311, 2821763.831232311)

    test_x, test_y = geotool.datum_compense(xc=x, yc=y, hb=400)

    assert test_x == pytest.approx(expected_x)
    assert test_y == pytest.approx(expected_y)
