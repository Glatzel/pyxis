import pytest

import pyxis


def test_lbh2xyz():
    x, y, z = pyxis.lbh2xyz(48.8566, 2.3522, 35.0)
    assert x == pytest.approx(4192979.6198897623)
    assert y == pytest.approx(4799159.563725418)
    assert z == pytest.approx(260022.66015989496)


def test_xyz2lbh():
    lon, lat, h = pyxis.xyz2lbh(4192979.6198897623, 4799159.563725418, 260022.66015989496)
    assert lon == pytest.approx(48.8566)
    assert lat == pytest.approx(2.3522)
    assert h == pytest.approx(35, abs=1e-3)
