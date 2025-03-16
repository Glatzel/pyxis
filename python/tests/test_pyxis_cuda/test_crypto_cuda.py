from copy import deepcopy

import numpy as np
import pytest

import pyxis
import pyxis.pyxis_cuda
from pyxis import COORD_CRYPTO_SPACE

bd09 = (121.10271724622564, 30.61484575976839)
gcj02 = (121.09626927850977, 30.608604331756705)
wgs84 = (121.0917077, 30.6107779)


bd09_array = (
    np.array([121.10271724622564, 121.10271724622564], np.float64),
    np.array([30.61484575976839, 30.61484575976839], np.float64),
)
gcj02_array = (
    np.array([121.09626927850977, 121.09626927850977], np.float64),
    np.array([30.608604368560773, 30.608604368560773], np.float64),
)
wgs84_array = (
    np.array([121.0917077, 121.0917077], np.float64),
    np.array([30.6107779, 30.6107779], np.float64),
)


# @pytest.mark.parametrize(
#     ("src", "dst", "input", "expected"),
#     [
#         (COORD_CRYPTO_SPACE.WGS84, COORD_CRYPTO_SPACE.BD09, wgs84_array, bd09),
#         (COORD_CRYPTO_SPACE.WGS84, COORD_CRYPTO_SPACE.GCJ02, wgs84_array, gcj02),
#         (COORD_CRYPTO_SPACE.GCJ02, COORD_CRYPTO_SPACE.BD09, gcj02_array, bd09),
#         (COORD_CRYPTO_SPACE.GCJ02, COORD_CRYPTO_SPACE.WGS84, gcj02_array, wgs84),
#         (COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.GCJ02, bd09_array, gcj02),
#         (COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.WGS84, bd09_array, wgs84),
#     ],
# )
# @pytest.mark.parametrize("exact", [True, False])
# def test_simple(src, dst, input, expected, exact):
#     import cupy as cp

#     in_lon = deepcopy(input[0])
#     in_lat = deepcopy(input[1])
#     module = pyxis.pyxis_cuda.CryptoCuda()
#     lon, lat = module.crypto_exact("double", cp.asarray(in_lon), cp.asarray(in_lat), src, dst, 1e-17, 100)
#     assert cp.asnumpy(lon)[0] == pytest.approx(expected[0])
#     assert cp.asnumpy(lat)[0] == pytest.approx(expected[1])


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("src", "dst", "input", "expected"),
    [
        (COORD_CRYPTO_SPACE.GCJ02, COORD_CRYPTO_SPACE.WGS84, gcj02_array, wgs84),
        (COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.GCJ02, bd09_array, gcj02),
        (COORD_CRYPTO_SPACE.BD09, COORD_CRYPTO_SPACE.WGS84, bd09_array, wgs84),
    ],
)
def test_exact(src, dst, input, expected):
    import cupy as cp

    in_lon = cp.asarray(deepcopy(input[0]), cp.float64)
    in_lat = cp.asarray(deepcopy(input[1]), cp.float64)

    module = pyxis.pyxis_cuda.CryptoCuda()
    module.crypto_exact("double", in_lon, in_lat, src, dst, 1e-17, 100)
    assert cp.asnumpy(in_lon)[0] == pytest.approx(expected[0])
    assert cp.asnumpy(in_lat)[0] == pytest.approx(expected[1])
