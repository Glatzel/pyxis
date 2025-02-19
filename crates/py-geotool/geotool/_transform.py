from typing import overload

import numpy as np
from numpy.typing import NDArray

from ._utils import TCoordArray, TCoordScalar, coord_util


@overload
def datum_compense(
    xc: TCoordScalar,
    yc: TCoordScalar,
    hb: float,
    radius: float = 6378_137,
    x0: float = 500_000,
    y0: float = 0,
) -> tuple[float, float]: ...
@overload
def datum_compense(
    xc: TCoordArray,
    yc: TCoordArray,
    hb: float,
    radius: float = 6378_137,
    x0: float = 500_000,
    y0: float = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def datum_compense(
    xc,
    yc,
    hb: float,
    radius: float = 6378_137,
    x0: float = 500_000,
    y0: float = 0,
):
    r"""
    Convert projected XY coordinates from height compensation plane to sea level plane.

    Unit: meter

    Parameters
    ----------
    xc, yc
        Coordinates on height compensation plane.
    hb
        Elevation of height compensation plane.
    radius
        Radius of earth.
    x0, y0
        Coordinate system origin.

    Returns
    -------
    tuple[Any, Any]
        Projected XY coordinates of sea level plane.

    References
    ----------
    .. [1] 杨元兴.抵偿高程面的选择与计算[J].城市勘测,2008(02):72-74.

    Examples
    --------
    >>> from glatzel import geotool
    >>> geotool.datum_compense(469704.6693, 2821940.796, 400)
    2821763.831232311 469706.56912942487
    >>> geotool.datum_compense(
    ...     np.array([469704.6693, 469704.6693]),
    ...     np.array([2821940.796, 2821940.796]),
    ...     400,
    ... )
    ( array([469706.56912942, 469706.56912942]), array([2821763.83123231, 2821763.83123231]),)
    """
    from .py_geotool import py_datum_compense  # type: ignore

    xc = coord_util("xc", xc)
    yc = coord_util("yc", yc)

    xc, yc = py_datum_compense(xc, yc, hb, radius, x0, y0)

    return xc, yc


@overload
def lbh2xyz(
    lon: TCoordScalar,
    lat: TCoordScalar,
    height: TCoordScalar,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
) -> tuple[float, float, float]: ...
@overload
def lbh2xyz(
    lon: TCoordArray,
    lat: TCoordArray,
    height: TCoordArray,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def lbh2xyz(
    lon,
    lat,
    height,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
):
    """
    Convert geodetic coordinates (longitude/L, latitude/B, height/H) to Cartesian coordinates (X, Y, Z).

    Parameters
    ----------
    lon
        Geodetic longitude(s) in degrees. Can be a single value or an array of values.
    lat
        Geodetic latitude(s) in degrees. Can be a single value or an array of values.
    height
        Ellipsoidal height(s) in meters. Can be a single value or an array of values.
    major_radius
        Semi major axis.
    invf
        Inverse flattening.

    Returns
    -------
    X : float or ndarray
        Cartesian X-coordinate(s) in meters. Same shape as the input latitude, longitude, and height.
    Y : float or ndarray
        Cartesian Y-coordinate(s) in meters. Same shape as the input latitude, longitude, and height.
    Z : float or ndarray
        Cartesian Z-coordinate(s) in meters. Same shape as the input latitude, longitude, and height.

    Notes
    -----
    - The conversion uses the WGS84 ellipsoid parameters:
        - Semi-major axis : 6378137.0 meters
        - Inverse Flattening :  298.257223563
    - Latitude and longitude should be provided in degrees and are internally converted to radians.

    Examples
    --------
    Convert a single geodetic coordinate:

    >>> latitude = 48.8566
    >>> longitude = 2.3522
    >>> height = 35
    >>> x, y, z = lbh2xyz(latitude, longitude, height, 6378137.0, 298.257223563)
    >>> print(f"X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f}")

    Convert multiple geodetic coordinates:

    >>> latitudes = np.array([48.8566, 51.5074])
    >>> longitudes = np.array([2.3522, -0.1278])
    >>> heights = np.array([35, 45])
    >>> x, y, z = lbh2xyz(latitudes, longitudes, heights, 6378137.0, 298.257223563)
    >>> print(f"X: {x}, Y: {y}, Z: {z}")
    """
    from .py_geotool import py_lbh2xyz  # type: ignore

    lon = coord_util("lon", lon)
    lat = coord_util("lat", lat)
    height = coord_util("height", height)

    x, y, z = py_lbh2xyz(lon, lat, height, major_radius, invf)

    return x, y, z


@overload
def xyz2lbh(
    x: TCoordScalar,
    y: TCoordScalar,
    z: TCoordScalar,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
    threshold: float = 1e-17,
    max_iter: int = 100,
) -> tuple[float, float, float]: ...
@overload
def xyz2lbh(
    x: TCoordArray,
    y: TCoordArray,
    z: TCoordArray,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
    threshold: float = 1e-17,
    max_iter: int = 100,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def xyz2lbh(
    x,
    y,
    z,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
    threshold: float = 1e-17,
    max_iter: int = 100,
):
    """
    Convert Cartesian coordinates (X, Y, Z) to geodetic coordinates (Longitude, Latitude, Height).

    Parameters
    ----------
    x : float or ndarray
        X coordinate(s) in meters.
    y : float or ndarray
        Y coordinate(s) in meters.
    z : float or ndarray
        Z coordinate(s) in meters.
    major_radius
        Semi major axis.
    invf
        Inverse flattening.
    threshold
        Error threshold.
    max_iter
        Max iterations.

    Returns
    -------
    longitude : float or ndarray
        Longitude in degrees.
    latitude : float or ndarray
        Latitude in degrees.
    height : float or ndarray
        Height above the reference ellipsoid in meters.

    Notes
    -----
    The function assumes the WGS84 ellipsoid for the conversion.

    Examples
    --------
    >>> x, y, z = 4510731.0, 4510731.0, 4510731.0
    >>> longitude, latitude, height = xyz_to_lbh(x, y, z)
    >>> print(f"Longitude: {longitude:.6f}°")
    >>> print(f"Latitude: {latitude:.6f}°")
    >>> print(f"Height: {height:.2f} m")
    """
    from .py_geotool import py_xyz2lbh  # type: ignore

    x = coord_util("x", x)
    y = coord_util("y", y)
    z = coord_util("z", z)

    lon, lat, h = py_xyz2lbh(x, y, z, major_radius, invf, threshold, max_iter)

    return lon, lat, h
