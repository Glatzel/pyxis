import geotool
import pytest

wgs84 = (121.09170737767907, 30.6107684401777)
gcj02 = (121.09626895522744, 30.608594904771056)
bd09 = (121.10271691314193, 30.614836298418275)


@pytest.mark.parametrize(
    ("func", "input", "expected"),
    [
        (geotool.wgs84_to_bd09, wgs84, bd09),
        (geotool.wgs84_to_gcj02, wgs84, gcj02),
        (geotool.gcj02_to_bd09, gcj02, bd09),
        (geotool.gcj02_to_wgs84, gcj02, wgs84),
        (geotool.bd09_to_gcj02, bd09, gcj02),
        (geotool.bd09_to_wgs84, bd09, wgs84),
    ],
)
def test_geotool(func, input, expected):
    lon, lat = func(*input)

    assert lon == pytest.approx(expected[0])
    assert lat == pytest.approx(expected[1])


@pytest.mark.parametrize(
    ("src", "dst", "input", "expected"),
    [
        ("WGS84", "BD09", wgs84, bd09),
        ("WGS84", "GCJ02", wgs84, gcj02),
        ("GCJ02", "BD09", gcj02, bd09),
        ("GCJ02", "WGS84", gcj02, wgs84),
        ("BD09", "GCJ02", bd09, gcj02),
        ("BD09", "WGS84", bd09, wgs84),
        pytest.param("WGS84", "WGS84", bd09, wgs84, marks=pytest.mark.xfail(strict=True)),
        pytest.param("cgcs", "WGS84", bd09, wgs84, marks=pytest.mark.xfail(strict=True)),
    ],
)
def test_convert(src, dst, input, expected):
    lon, lat = geotool.crypto(input[0], input[1], src, dst)

    assert lon == pytest.approx(expected[0])
    assert lat == pytest.approx(expected[1])
