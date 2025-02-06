import geotool
import pytest


@pytest.mark.parametrize(
    ("x", "y", "z", "r1", "u1", "z1"),
    [
        (
            1.2,
            3.4,
            -5.6,
            3.60555127546399,
            1.23150371234085,
            -5.60000000000000,
        ),
    ],
)
def test_cartesian_to_cylindrical(x, y, z, r1, u1, z1):
    cy = geotool.cartesian_to_cylindrical(x=x, y=y, z=z)
    assert cy[0] == pytest.approx(r1)
    assert cy[1] == pytest.approx(u1)
    assert cy[2] == pytest.approx(z1)


@pytest.mark.parametrize(
    ("x", "y", "z", "u2", "v2", "r2"),
    [
        (
            1.2,
            3.4,
            -5.6,
            1.23150371234085,
            2.5695540653144073,
            6.66033032213868,
        ),
    ],
)
def test_cartesian_to_spherical(x, y, z, u2, v2, r2):
    cy = geotool.cartesian_to_spherical(x=x, y=y, z=z)
    assert cy[0] == pytest.approx(u2)
    assert cy[1] == pytest.approx(v2)
    assert cy[2] == pytest.approx(r2)


@pytest.mark.parametrize(
    ("x", "y", "z", "r1", "u1", "z1"),
    [
        (
            1.2,
            3.4,
            -5.6,
            3.60555127546399,
            1.23150371234085,
            -5.60000000000000,
        ),
    ],
)
def test_cylindrical_to_cartesian(x, y, z, r1, u1, z1):
    cy = geotool.cylindrical_to_cartesian(r1, u1, z1)
    assert cy[0] == pytest.approx(x)
    assert cy[1] == pytest.approx(y)
    assert cy[2] == pytest.approx(z)


@pytest.mark.parametrize(
    ("r1", "u1", "z1", "u2", "v2", "r2"),
    [
        (
            3.60555127546399,
            1.23150371234085,
            -5.60000000000000,
            1.23150371234085,
            2.5695540653144073,
            6.66033032213868,
        ),
    ],
)
def test_cylindrical_to_spherical(r1, u1, z1, u2, v2, r2):
    cy = geotool.cylindrical_to_spherical(r1, u1, z1)
    assert cy[0] == pytest.approx(u2)
    assert cy[1] == pytest.approx(v2)
    assert cy[2] == pytest.approx(r2)


@pytest.mark.parametrize(
    ("x", "y", "z", "u2", "v2", "r2"),
    [
        (
            1.2,
            3.4,
            -5.6,
            1.23150371234085,
            2.5695540653144073,
            6.66033032213868,
        ),
    ],
)
def test_spherical_to_cartesian(x, y, z, u2, v2, r2):
    cy = geotool.spherical_to_cartesian(u2, v2, r2)
    assert cy[0] == pytest.approx(x)
    assert cy[1] == pytest.approx(y)
    assert cy[2] == pytest.approx(z)


@pytest.mark.parametrize(
    ("r1", "u1", "z1", "u2", "v2", "r2"),
    [
        (
            3.60555127546399,
            1.23150371234085,
            -5.60000000000000,
            1.23150371234085,
            2.5695540653144073,
            6.66033032213868,
        ),
    ],
)
def test_spherical_to_cylindrical(r1, u1, z1, u2, v2, r2):
    cy = geotool.spherical_to_cylindrical(u2, v2, r2)
    assert cy[0] == pytest.approx(r1)
    assert cy[1] == pytest.approx(u1)
    assert cy[2] == pytest.approx(z1)
