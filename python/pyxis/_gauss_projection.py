from typing import overload

import numpy as np
from numpy.typing import NDArray

from ._utils import TCoordArray, TCoordScalar, coord_util


@overload
def lbh2xyz(
    lon: TCoordScalar,
    lat: TCoordScalar,
    height: TCoordScalar,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
    clone: bool = True,
) -> tuple[float, float, float]: ...
@overload
def lbh2xyz(
    lon: TCoordArray,
    lat: TCoordArray,
    height: TCoordArray,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
    clone: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def lbh2xyz(
    lon,
    lat,
    height,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
    clone: bool = True,
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
    clone
        If True, a deep copy of `data` will be created before processing.
        If False, `data` will be modified in place. Default is False.

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
    """
    from .pyxis_py import py_lbh2xyz  # type: ignore

    lon = coord_util("lon", lon, clone)
    lat = coord_util("lat", lat, clone)
    height = coord_util("height", height, clone)

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
    clone: bool = True,
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
    clone: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def xyz2lbh(
    x,
    y,
    z,
    major_radius: float = 6378137.0,
    invf: float = 298.257223563,
    threshold: float = 1e-17,
    max_iter: int = 100,
    clone: bool = True,
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
    clone
        If True, a deep copy of `data` will be created before processing.
        If False, `data` will be modified in place. Default is False.

    Returns
    -------
    longitude : float or ndarray
        Longitude in degrees.
    latitude : float or ndarray
        Latitude in degrees.
    height : float or ndarray
        Height above the reference ellipsoid in meters.
    clone
        If True, a deep copy of `data` will be created before processing.
        If False, `data` will be modified in place. Default is False.

    Notes
    -----
    The function assumes the WGS84 ellipsoid for the conversion.
    """
    from .pyxis_py import py_xyz2lbh  # type: ignore

    x = coord_util("x", x, clone)
    y = coord_util("y", y, clone)
    z = coord_util("z", z, clone)

    lon, lat, h = py_xyz2lbh(x, y, z, major_radius, invf, threshold, max_iter)

    return lon, lat, h
