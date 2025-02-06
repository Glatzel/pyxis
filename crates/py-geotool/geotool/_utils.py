from typing import Union, overload

import numpy as np
from numpy.typing import NDArray

TCoordScalar = Union[int, float, np.integer, np.floating]
TCoordArray = Union[list[float], tuple[float, ...], NDArray]
TCoord = Union[TCoordScalar, TCoordArray]


@overload
def coord_util(name: str, coord: TCoordScalar) -> float: ...
@overload
def coord_util(name: str, coord: TCoordArray) -> NDArray[np.float64]: ...
def coord_util(name: str, coord: TCoord) -> float | NDArray[np.float64]:
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
