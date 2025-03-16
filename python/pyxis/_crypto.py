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
        Coordinates of `BD09` coordinate system.
    crypto_from, crypto_to
        From a coordinate system to another coordinate system.
    exact
        Use exact mode. Max Error is 1e-13.
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

    Examples
    --------
    >>> import pyxis
    >>> pyxis.CoordCrypto.convert(
    ...     121.09170737767907, 30.6107684401777, "WGS84", "GCJ02"
    ... )
    (121.09626895522744, 30.608594904771056)
    """
    from .pyxis_py import py_crypto  # type: ignore

    lon = coord_util("lon", lon, clone)
    lat = coord_util("lat", lat, clone)
    if (
        (str(crypto_from).upper() not in COORD_CRYPTO_SPACE.list())
        or (str(crypto_to).upper() not in COORD_CRYPTO_SPACE.list())
        or str(crypto_to).upper() == str(crypto_from).upper()
    ):
        msg = f"Unsupported: from `{crypto_from}` to `{crypto_to}`."
        raise TypeError(msg)
    lon, lat = py_crypto(lon, lat, crypto_from, crypto_to, exact)

    return lon, lat
