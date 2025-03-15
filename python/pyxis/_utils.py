import copy
from typing import overload

import numpy as np
from numpy.typing import NDArray

TCoordScalar = int | float | np.integer | np.floating
TCoordArray = list[float] | tuple[float, ...] | NDArray
TCoord = TCoordScalar | TCoordArray


@overload
def coord_util(name: str, coord: TCoordScalar, clone: bool) -> float: ...
@overload
def coord_util(name: str, coord: TCoordArray, clone: bool) -> NDArray[np.float64]: ...
def coord_util(name: str, coord: TCoord, clone: bool) -> float | NDArray[np.float64]:
    coord = copy.deepcopy(coord) if clone else coord
    match coord:
        case float():
            pass
        case int() | np.integer() | np.floating():
            coord = float(coord)
        case list() | tuple() | np.ndarray():
            coord = np.asarray(coord, np.float64)
        case _:
            msg = f"Unsupported type: {name} {type(coord)}"
            raise TypeError(msg)
    return coord
