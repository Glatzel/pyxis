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
    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)
    match crypto_from, crypto_to:
        case "WGS84", "GCJ02":
            result_lon, result_lat = wgs84_to_gcj02(lon, lat)
        case "GCJ02", "WGS84":
            result_lon, result_lat = gcj02_to_wgs84(lon, lat)
        case "WGS84", "BD09":
            result_lon, result_lat = wgs84_to_bd09(lon, lat)
        case "BD09", "WGS84":
            result_lon, result_lat = bd09_to_wgs84(lon, lat)
        case "GCJ02", "BD09":
            result_lon, result_lat = gcj02_to_bd09(lon, lat)
        case "BD09", "GCJ02":
            result_lon, result_lat = bd09_to_gcj02(lon, lat)
        case _:
            msg = f"from `{crypto_from}` to `{crypto_to}`."
            raise TypeError(msg)

    return result_lon, result_lat


@overload
def bd09_to_gcj02(
    lon: TCoordScalar,
    lat: TCoordScalar,
) -> tuple[float, float]: ...
@overload
def bd09_to_gcj02(
    lon: TCoordArray,
    lat: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def bd09_to_gcj02(lon, lat):
    r"""
    Convert coordinates from `BD09` to `GCJ02`.

    Parameters
    ----------
    lon, lat
        Coordinates of `BD09` coordinate system.

    Returns
    -------
    tuple[Any, Any]
        Longitude and latitude of `GCJ02` coordinate system.

    Examples
    --------
    >>> from glatzel import geotool
    >>> geotool.CoordCrypto.bd09_to_gcj02(121.10271691314193, 30.614836298418275)
    (121.09626895522744, 30.608594904771056)
    """
    from .py_geotool import py_bd09_to_gcj02  # type: ignore

    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)

    lon, lat = py_bd09_to_gcj02(lon, lat)

    return lon, lat


@overload
def bd09_to_wgs84(
    lon: TCoordScalar,
    lat: TCoordScalar,
) -> tuple[float, float]: ...
@overload
def bd09_to_wgs84(
    lon: TCoordArray,
    lat: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def bd09_to_wgs84(lon, lat):
    r"""
    Convert coordinates from `BD09` to `WGS84`.

    Parameters
    ----------
    lon, lat
        Coordinates of `BD09` coordinate system.

    Returns
    -------
    tuple[Any, Any]
        Longitude and latitude of `WGS84` coordinate system.

    Examples
    --------
    >>> from glatzel import geotool
    >>> geotool.CoordCrypto.bd09_to_wgs84(121.10271691314193, 30.614836298418275)
    (121.09170737767907, 30.6107684401777)
    """
    from .py_geotool import py_bd09_to_wgs84  # type: ignore

    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)

    lon, lat = py_bd09_to_wgs84(lon, lat)

    return lon, lat


@overload
def gcj02_to_bd09(
    lon: TCoordScalar,
    lat: TCoordScalar,
) -> tuple[float, float]: ...
@overload
def gcj02_to_bd09(
    lon: TCoordArray,
    lat: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def gcj02_to_bd09(lon, lat):
    r"""
    Convert coordinates from `GCJ02` to `BD09`.

    Parameters
    ----------
    lon, lat
        Coordinates of `GCJ02` coordinate system.

    Returns
    -------
    tuple[Any, Any]
        Longitude and latitude of `BD09` coordinate system.

    Examples
    --------
    >>> from glatzel import geotool
    >>> geotool.CoordCrypto.gcj02_to_bd09(121.09626895522744, 30.608594904771056)
    (121.10271691314193, 30.614836298418275)
    """
    from .py_geotool import py_gcj02_to_bd09  # type: ignore

    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)

    lon, lat = py_gcj02_to_bd09(lon, lat)

    return lon, lat


@overload
def gcj02_to_wgs84(
    lon: TCoordScalar,
    lat: TCoordScalar,
) -> tuple[float, float]: ...
@overload
def gcj02_to_wgs84(
    lon: TCoordArray,
    lat: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def gcj02_to_wgs84(lon, lat):
    r"""
    Convert coordinates from `GCJ02` to `WGS84`.

    Parameters
    ----------
    lon, lat
        Coordinates of `GCJ02` coordinate system.

    Returns
    -------
    tuple[Any, Any]
        Longitude and latitude of `WGS84` coordinate system.

    Examples
    --------
    >>> from glatzel import geotool
    >>> geotool.CoordCrypto.gcj02_to_wgs84(121.09626895522744, 30.608594904771056)
    (121.09170737767907, 30.6107684401777)
    """
    from .py_geotool import py_gcj02_to_wgs84  # type: ignore

    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)

    lon, lat = py_gcj02_to_wgs84(lon, lat)

    return lon, lat


@overload
def wgs84_to_gcj02(
    lon: TCoordScalar,
    lat: TCoordScalar,
) -> tuple[float, float]: ...
@overload
def wgs84_to_gcj02(
    lon: TCoordArray,
    lat: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def wgs84_to_gcj02(lon, lat):
    r"""
    Convert coordinates from `WGS84` to `GCJ02`.

    Parameters
    ----------
    lon, lat
        Coordinates of `WGS84` coordinate system.

    Returns
    -------
    tuple[float,float]
        Longitude and latitude of `GCJ02` coordinate system.

    Examples
    --------
    >>> from glatzel import geotool
    >>> geotool.CoordCrypto.wgs84_to_gcj02(121.09170737767907, 30.6107684401777)
    (121.09626895522744, 30.608594904771056)
    """
    from .py_geotool import py_wgs84_to_gcj02  # type: ignore

    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)

    lon, lat = py_wgs84_to_gcj02(lon, lat)

    return lon, lat


@overload
def wgs84_to_bd09(
    lon: TCoordScalar,
    lat: TCoordScalar,
) -> tuple[float, float]: ...
@overload
def wgs84_to_bd09(
    lon: TCoordArray,
    lat: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def wgs84_to_bd09(lon, lat):
    r"""
    Convert coordinates from `WGS84` to `BD09`.

    Parameters
    ----------
    lon, lat
        Coordinates of `WGS84` coordinate system.

    Returns
    -------
    tuple[Any, Any]
        Longitude and latitude of `BD09` coordinate system.

    Examples
    --------
    >>> from glatzel import geotool
    >>> geotool.CoordCrypto.wgs84_to_bd09(121.09170737767907, 30.6107684401777)
    (121.10271691314193, 30.614836298418275)
    """
    from .py_geotool import py_wgs84_to_bd09  # type: ignore

    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)

    lon, lat = py_wgs84_to_bd09(lon, lat)

    return lon, lat
