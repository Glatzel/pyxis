import numpy as np
import pytest

import pyxis


def test_datum_compense_scalar():
    expected_x, expected_y = 469706.56912942487, 2821763.831232311
    test_x, test_y = pyxis.datum_compense(xc=469704.6693, yc=2821940.796, hb=400)

    assert test_x == pytest.approx(expected_x)
    assert test_y == pytest.approx(expected_y)


@pytest.mark.parametrize(
    ("x", "y"),
    [
        pytest.param(
            np.array([469704.6693], np.float64),
            np.array([2821940.796], np.float64),
            id="ndarry1-ndarry1",
        ),
        pytest.param(
            np.array([469704.6693, 469704.6693], np.float64),
            np.array([2821940.796, 2821940.796], np.float64),
            id="ndarry2-ndarry2",
        ),
    ],
)
def test_datum_compense_vector(x, y):
    expected_x = 469706.56912942487
    expected_y = 2821763.831232311

    test_x, test_y = pyxis.datum_compense(xc=x, yc=y, hb=400)

    assert test_x[0] == pytest.approx(expected_x)
    assert test_y[0] == pytest.approx(expected_y)
