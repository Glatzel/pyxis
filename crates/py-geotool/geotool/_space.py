from enum import Enum
from typing import overload

import numpy as np
from numpy.typing import NDArray

from ._utils import TCoordArray, TCoordScalar, coord_util


class COORD_SPACE(str, Enum):
    """COORD_SPACE."""

    CARTESIAN = "CARTESIAN"
    CYLINDRICAL = "CYLINDRICAL"
    SPHERICAL = "SPHERICAL"


@overload
def space(
    x: TCoordScalar,
    y: TCoordScalar,
    z: TCoordScalar,
    from_space: COORD_SPACE,
    to_space: COORD_SPACE,
) -> tuple[float, float, float]: ...
@overload
def space(
    x: TCoordArray,
    y: TCoordArray,
    z: TCoordArray,
    from_space: COORD_SPACE,
    to_space: COORD_SPACE,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def space(x, y, z, from_space, to_space):
    from .py_geotool import py_space  # type: ignore

    if (
        (str(from_space).lower() not in ("wgs84", "bd09", "gcj02"))
        or (str(to_space).lower() not in ("wgs84", "bd09", "gcj02"))
        or str(to_space).lower() == str(from_space).lower()
    ):
        msg = f"from `{from_space}` to `{to_space}`."
        raise TypeError(msg)

    x = coord_util("x", x)
    y = coord_util("y", y)
    z = coord_util("z", z)

    r, u, z = py_space(x, y, z, "cartesian", "cylindrical")

    return r, u, z
