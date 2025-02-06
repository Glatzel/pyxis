import geotool
import pytest


def test_angle2dms():
    d, m, s = geotool.angle2dms(angle=30.76)
    assert int(d) == 30
    assert int(m) == 45
    assert int(s) == 36


def test_dms2angle():
    a = geotool.dms2angle(deg=30, min=45, sec=36)
    assert a == pytest.approx(30.76)
