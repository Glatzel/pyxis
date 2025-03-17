from typing import Any

import numpy as np


def angle2dms(angle) -> tuple[Any, Any, Any]:
    """
    Convert float angle to degree, minute, second.

    Parameters
    ----------
    angle
        Float angle.

    Returns
    -------
    tuple[Any,Any,Any]
        Degree, minute, second.
    """
    deg = np.trunc(angle)
    min = np.trunc((angle - deg) * 60.0)
    sec = (angle - deg - min / 60.0) * 3600.0
    return deg, min, sec


def dms2angle(deg, min, sec):
    """
    Convert degree, minute, second to float angle.

    Parameters
    ----------
    deg, min, sec
        Degree, minute, second.

    Returns
    -------
    Any
        Float Angle.

    Examples
    --------
    >>> from glatzel import pyxis
    >>> pyxis.dms2angle(30, 45, 36)
    30.76
    """
    return deg + min / 60.0 + sec / 3600.0
