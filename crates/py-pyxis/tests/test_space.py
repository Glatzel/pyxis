import pyxis
import pytest
from pyxis import COORD_SPACE

cartesian = (1.2, 3.4, -5.6)
cylindrical = (3.60555127546399, 1.23150371234085, -5.60000000000000)
spherical = (1.23150371234085, 2.5695540653144073, 6.66033032213868)


@pytest.mark.parametrize(
    ("from_space", "to_space", "input", "expected"),
    [
        (COORD_SPACE.CARTESIAN, COORD_SPACE.CYLINDRICAL, cartesian, cylindrical),
        (COORD_SPACE.CARTESIAN, COORD_SPACE.SPHERICAL, cartesian, spherical),
        (COORD_SPACE.CYLINDRICAL, COORD_SPACE.CARTESIAN, cylindrical, cartesian),
        (COORD_SPACE.CYLINDRICAL, COORD_SPACE.SPHERICAL, cylindrical, spherical),
        (COORD_SPACE.SPHERICAL, COORD_SPACE.CARTESIAN, spherical, cartesian),
        (COORD_SPACE.SPHERICAL, COORD_SPACE.CYLINDRICAL, spherical, cylindrical),
    ],
)
def test_space(from_space, to_space, input, expected):
    result = pyxis.space(input[0], input[1], input[2], from_space, to_space)
    assert result[0] == pytest.approx(expected[0])
    assert result[1] == pytest.approx(expected[1])
    assert result[2] == pytest.approx(expected[2])
