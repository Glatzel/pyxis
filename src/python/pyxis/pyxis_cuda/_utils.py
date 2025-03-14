import copy
from pathlib import Path
from typing import Literal, Union

import cupy as cp
import numpy as np
from numpy.typing import NDArray

PTX_PATH = Path(__file__).parent / "ptx"
TDTYPE = Literal["float32", "float64"]
TCoordScalarCuda = Union[int, float, np.integer, np.floating]
TCoordArrayCuda = Union[list[float], tuple[float, ...], NDArray]


def scalar_util(scalar_like: TCoordScalarCuda, dtype: TDTYPE) -> cp.ndarray:
    scalar_like = copy.deepcopy(scalar_like)
    scalar_like = cp.float32(scalar_like) if dtype == "float32" else cp.float64(scalar_like)
    return scalar_like


def array_util(array_like: TCoordArrayCuda, dtype: TDTYPE) -> cp.ndarray:
    array_like = copy.deepcopy(array_like)
    array_like = cp.asarray(array_like, cp.float32 if dtype == "float32" else cp.float64)

    return array_like
