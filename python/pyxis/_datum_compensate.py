from typing import overload

import numpy as np
from numpy.typing import NDArray

from ._utils import TCoordArray, TCoordScalar, coord_util


@overload
def datum_compensate(
    xc: TCoordScalar,
    yc: TCoordScalar,
    hb: float,
    radius: float,
    x0: float,
    y0: float,
    clone: bool = True,
) -> tuple[float, float]: ...
@overload
def datum_compensate(
    xc: TCoordArray,
    yc: TCoordArray,
    hb: float,
    radius: float,
    x0: float,
    y0: float,
    clone: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def datum_compensate(
    xc,
    yc,
    hb: float,
    radius: float,
    x0: float,
    y0: float,
    clone: bool = True,
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
    clone
        If True, a deep copy of `data` will be created before processing.
        If False, `data` will be modified in place. Default is False.

    Returns
    -------
    tuple[Any, Any]
        Projected XY coordinates of sea level plane.

    References
    ----------
    - 杨元兴.抵偿高程面的选择与计算[J].城市勘测,2008(02):72-74.
    """
    from .pyxis_py import py_datum_compensate  # type: ignore

    xc = coord_util("xc", xc, clone)
    yc = coord_util("yc", yc, clone)

    xc, yc = py_datum_compensate(xc, yc, hb, radius, x0, y0)

    return xc, yc
