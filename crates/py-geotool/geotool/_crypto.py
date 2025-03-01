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


@overload
def crypto(
    lon: TCoordScalar,
    lat: TCoordScalar,
    crypto_from: COORD_CRYPTO_SPACE,
    crypto_to: COORD_CRYPTO_SPACE,
) -> tuple[float, float]: ...
@overload
def crypto(
    lon: TCoordArray,
    lat: TCoordArray,
    crypto_from: COORD_CRYPTO_SPACE,
    crypto_to: COORD_CRYPTO_SPACE,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def crypto(lon, lat, crypto_from, crypto_to):
    r"""
    Convert coordinates between `WGS84`, `GCJ02` and `BD09`.

    Parameters
    ----------
    lon, lat
        Coordinates of `BD09` coordinate system.
    crypto_from, crypto_to
        From a coordinate system to another coordinate system.

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
    >>> import geotool
    >>> geotool.CoordCrypto.convert(
    ...     121.09170737767907, 30.6107684401777, "WGS84", "GCJ02"
    ... )
    (121.09626895522744, 30.608594904771056)
    """
    from .py_geotool import py_crypto  # type: ignore

    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)
    if (
        (str(crypto_from).lower() not in ("wgs84", "bd09", "gcj02"))
        or (str(crypto_to).lower() not in ("wgs84", "bd09", "gcj02"))
        or str(crypto_to).lower() == str(crypto_from).lower()
    ):
        msg = f"from `{crypto_from}` to `{crypto_to}`."
        raise TypeError(msg)

    return py_crypto(lon, lat, crypto_from, crypto_to)
