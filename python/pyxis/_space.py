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

    @classmethod
    def list(cls):
        return list(map(str, cls))


@overload
def space(
    x: TCoordScalar,
    y: TCoordScalar,
    z: TCoordScalar,
    from_space: COORD_SPACE,
    to_space: COORD_SPACE,
    clone: bool = True,
) -> tuple[float, float, float]: ...
@overload
def space(
    x: TCoordArray,
    y: TCoordArray,
    z: TCoordArray,
    from_space: COORD_SPACE,
    to_space: COORD_SPACE,
    clone: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
def space(
    x,
    y,
    z,
    from_space,
    to_space,
    clone: bool = True,
):
    from .pyxis_py import py_space  # type: ignore

    if (
        (str(from_space).upper() not in COORD_SPACE.list())
        or (str(to_space).upper() not in COORD_SPACE.list())
        or str(to_space).upper() == str(from_space).lower()
    ):
        msg = f"from `{from_space}` to `{to_space}`."
        raise TypeError(msg)

    x = coord_util("x", x, clone)
    y = coord_util("y", y, clone)
    z = coord_util("z", z, clone)

    r, u, z = py_space(x, y, z, from_space, to_space)

    return r, u, z
