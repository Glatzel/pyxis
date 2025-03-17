import numpy as np
import pytest
from numpy import cos, radians, sin

import pyxis


class TestCoordMigrate:
    @pytest.mark.parametrize(("origin_x", "origin_y", "rel_x", "rel_y"), [(10, 20, 2, -1)])
    @pytest.mark.parametrize("angle", np.linspace(30, 390, 4))
    def test_abs(self, origin_x, origin_y, rel_x, rel_y, angle):
        test_x, test_y = pyxis.CoordMigrate.abs(
            origin_x=origin_x, origin_y=origin_y, rel_x=rel_x, rel_y=rel_y, angle=angle
        )
        expected_x = cos(radians(-angle)) * rel_x + sin(radians(-angle)) * rel_y + origin_x
        expected_y = -sin(radians(-angle)) * rel_x + cos(radians(-angle)) * rel_y + origin_y

        assert test_x == pytest.approx(expected_x)
        assert test_y == pytest.approx(expected_y)

    @pytest.mark.parametrize(("abs_x", "abs_y", "rel_x", "rel_y"), [(10, 20, 2, -1)])
    @pytest.mark.parametrize("angle", np.linspace(30, 390, 4))
    def test_origin(self, abs_x, abs_y, rel_x, rel_y, angle):
        test_x, test_y = pyxis.CoordMigrate.origin(abs_x=abs_x, abs_y=abs_y, rel_x=rel_x, rel_y=rel_y, angle=angle)
        expected_x = abs_x - (cos(radians(-angle)) * rel_x + sin(radians(-angle)) * rel_y)
        expected_y = abs_y - (-sin(radians(-angle)) * rel_x + cos(radians(-angle)) * rel_y)

        assert test_x == pytest.approx(expected_x)
        assert test_y == pytest.approx(expected_y)

    @pytest.mark.parametrize(("origin_x", "origin_y", "abs_x", "abs_y"), [(10, 20, 2, -1)])
    @pytest.mark.parametrize("angle", np.linspace(30, 390, 4))
    def test_rel(self, origin_x, origin_y, abs_x, abs_y, angle):
        test_x, test_y = pyxis.CoordMigrate.rel(
            origin_x=origin_x,
            origin_y=origin_y,
            abs_x=abs_x,
            abs_y=abs_y,
            angle=angle,
        )
        expected_x = cos(radians(angle)) * (abs_x - origin_x) + sin(radians(angle)) * (abs_y - origin_y)
        expected_y = -sin(radians(angle)) * (abs_x - origin_x) + cos(radians(angle)) * (abs_y - origin_y)

        assert test_x == pytest.approx(expected_x)
        assert test_y == pytest.approx(expected_y)
