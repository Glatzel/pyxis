from typing import overload

import numpy as np
from numpy.typing import NDArray

from ._utils import TCoordArray, TCoordScalar, coord_util


@overload
def cartesian_to_cylindrical(
    x: TCoordScalar,
    y: TCoordScalar,
    z: TCoordScalar,
) -> tuple[float, float, float]: ...
@overload
def cartesian_to_cylindrical(
    x: TCoordArray,
    y: TCoordArray,
    z: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def cartesian_to_cylindrical(x, y, z):
    from .py_geotool import py_cartesian_to_cylindrical  # type: ignore

    x = coord_util("x", x)
    y = coord_util("y", y)
    z = coord_util("z", z)

    r, u, z = py_cartesian_to_cylindrical(x, y, z)

    return r, u, z


@overload
def cartesian_to_spherical(
    x: TCoordScalar,
    y: TCoordScalar,
    z: TCoordScalar,
) -> tuple[float, float, float]: ...
@overload
def cartesian_to_spherical(
    x: TCoordArray,
    y: TCoordArray,
    z: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def cartesian_to_spherical(x, y, z):
    from .py_geotool import py_cartesian_to_spherical  # type: ignore

    x = coord_util("x", x)
    y = coord_util("y", y)
    z = coord_util("z", z)

    u, v, r = py_cartesian_to_spherical(x, y, z)

    return u, v, r


@overload
def cylindrical_to_cartesian(
    r: TCoordScalar,
    u: TCoordScalar,
    z: TCoordScalar,
) -> tuple[float, float, float]: ...
@overload
def cylindrical_to_cartesian(
    r: TCoordArray,
    u: TCoordArray,
    z: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def cylindrical_to_cartesian(r, u, z):
    from .py_geotool import py_cylindrical_to_cartesian  # type: ignore

    r = coord_util("r", r)
    u = coord_util("u", u)
    z = coord_util("z", z)

    x, y, z = py_cylindrical_to_cartesian(r, u, z)

    return x, y, z


@overload
def cylindrical_to_spherical(
    r: TCoordScalar,
    u: TCoordScalar,
    z: TCoordScalar,
) -> tuple[float, float, float]: ...
@overload
def cylindrical_to_spherical(
    r: TCoordArray,
    u: TCoordArray,
    z: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def cylindrical_to_spherical(r, u, z):
    from .py_geotool import py_cylindrical_to_spherical  # type: ignore

    r = coord_util("r", r)
    u = coord_util("u", u)
    z = coord_util("z", z)

    u, v, r = py_cylindrical_to_spherical(r, u, z)

    return u, v, r


@overload
def spherical_to_cartesian(
    u: TCoordScalar,
    v: TCoordScalar,
    r: TCoordScalar,
) -> tuple[float, float, float]: ...
@overload
def spherical_to_cartesian(
    u: TCoordArray,
    v: TCoordArray,
    r: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def spherical_to_cartesian(u, v, r):
    from .py_geotool import py_spherical_to_cartesian  # type: ignore

    u = coord_util("u", u)
    v = coord_util("v", v)
    r = coord_util("r", r)

    x, y, z = py_spherical_to_cartesian(u, v, r)

    return x, y, z


@overload
def spherical_to_cylindrical(
    u: TCoordScalar,
    v: TCoordScalar,
    r: TCoordScalar,
) -> tuple[float, float, float]: ...
@overload
def spherical_to_cylindrical(
    u: TCoordArray,
    v: TCoordArray,
    r: TCoordArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def spherical_to_cylindrical(u, v, r):
    from .py_geotool import py_spherical_to_cylindrical  # type: ignore

    u = coord_util("u", u)
    v = coord_util("v", v)
    r = coord_util("r", r)

    r, u, z = py_spherical_to_cylindrical(u, v, r)

    return r, u, z
