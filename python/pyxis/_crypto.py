from enum import Enum
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ._utils import TCoordArray, TCoordScalar, coord_util


class COORD_CRYPTO_SPACE(str, Enum):
    """COORD_CRYPTO_SPACE."""

    WGS84 = "WGS84"
    GCJ02 = "GCJ02"
    BD09 = "BD09"

    @classmethod
    def list(cls):
        return list(map(str, cls))


@overload
def crypto(
    lon: TCoordScalar,
    lat: TCoordScalar,
    crypto_from: COORD_CRYPTO_SPACE,
    crypto_to: COORD_CRYPTO_SPACE,
    exact: bool,
    clone: bool = True,
) -> tuple[float, float]: ...
@overload
def crypto(
    lon: TCoordArray,
    lat: TCoordArray,
    crypto_from: COORD_CRYPTO_SPACE,
    crypto_to: COORD_CRYPTO_SPACE,
    exact: bool,
    clone: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def crypto(
    lon,
    lat,
    crypto_from,
    crypto_to,
    exact,
    clone: bool = True,
):
    r"""
    Convert coordinates between `WGS84`, `GCJ02` and `BD09`.

    Parameters
    ----------
    lon, lat
        Input Coordinates.
    crypto_from, crypto_to
        From a coordinate system to another coordinate system.
    exact
        Use exact mode, which raises precision to 1e-13, but will be sightly slower.
        This parameter only works when conversion:
        - from `BD09` to `GCJ02`
        - from `BD09` to `WGS84`
        - from `GCJ02` to `WGS84`
    clone
        If True, a deep copy of `data` will be created before processing.
        If False, `data` will be modified in place. Default is False.

    Returns
    -------
    tuple[Any, Any]
        Longitude and latitude.

    Raises
    ------
    ParameterError
        If ``crypto_from`` == ``crypto_to``.
        If ``crypto_from`` not in ``COORD_CRYPTO_SPACE``.
        If ``crypto_to`` not in ``COORD_CRYPTO_SPACE``.
    """
    from .pyxis_py import py_crypto  # type: ignore

    lon = coord_util("lon", lon, clone)
    lat = coord_util("lat", lat, clone)
    # check if crypto_from and crypto_to is valid
    # check if crypto_from and crypto_to are not equal
    if (
        (str(crypto_from).upper() not in COORD_CRYPTO_SPACE.list())
        or (str(crypto_to).upper() not in COORD_CRYPTO_SPACE.list())
        or str(crypto_to).upper() == str(crypto_from).upper()
    ):
        msg = f"Unsupported: from `{crypto_from}` to `{crypto_to}`."
        raise TypeError(msg)
    lon, lat = py_crypto(lon, lat, crypto_from, crypto_to, exact)

    return lon, lat
