import numpy as np
import pytest
import pyxis
from pyxis import COORD_CRYPTO_SPACE


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (COORD_CRYPTO_SPACE.WGS84, COORD_CRYPTO_SPACE.BD09),
        (COORD_CRYPTO_SPACE.WGS84, COORD_CRYPTO_SPACE.GCJ02),
        (COORD_CRYPTO_SPACE.GCJ02, COORD_CRYPTO_SPACE.BD09),
        (COORD_CRYPTO_SPACE.GCJ02, COORD_CRYPTO_SPACE.WGS84),
        (COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.GCJ02),
        (COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.WGS84),
    ],
)
@pytest.mark.parametrize("exact", [True, False])
def test_convert(src, dst, exact, snapshot):
    result = pyxis.crypto(120, 30, src, dst, exact)
    assert result == snapshot(name=f"{src}-{dst}-{exact}")


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (COORD_CRYPTO_SPACE.WGS84, COORD_CRYPTO_SPACE.BD09),
        (COORD_CRYPTO_SPACE.WGS84, COORD_CRYPTO_SPACE.GCJ02),
        (COORD_CRYPTO_SPACE.GCJ02, COORD_CRYPTO_SPACE.BD09),
        (COORD_CRYPTO_SPACE.GCJ02, COORD_CRYPTO_SPACE.WGS84),
        (COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.GCJ02),
        (COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.WGS84),
    ],
)
@pytest.mark.parametrize("exact", [True, False])
def test_convert_array(src, dst, exact, snapshot):
    result = pyxis.crypto(np.array([120, 121]), np.array([30, 31]), src, dst, exact)
    assert result == snapshot(name=f"{src}-{dst}-{exact}")
