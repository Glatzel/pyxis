from typing import Any

import numpy as np
from numpy import cos, deg2rad, sin


class CoordMigrate:
    r"""Convert coordinates between `rel-coords`, `absolute-coords` and `relative-origin`."""

    @classmethod
    def rotate_matrix(cls, angle: float):
        r"""
        Rotation matrix.

        Parameters
        ----------
        angle
            Angle of rotation measured **anticlockwise** in degree.

        Returns
        -------
        NDArray[np.float64]
            Rotate Matrix.

        Notes
        -----
        Rotate Matrix:

        .. math::

            \begin{bmatrix}
                cos(\angle) & sin(\angle) \\
                -sin(\angle) & cos(\angle)
            \end{bmatrix}
        """
        angle = deg2rad(angle)
        temp = [
            [cos(angle), sin(angle)],
            [-sin(angle), cos(angle)],
        ]
        return np.array(temp, dtype=np.float64)

    @classmethod
    def rel(
        cls,
        origin_x: float,
        origin_y: float,
        abs_x,
        abs_y,
        angle: float,
    ) -> tuple[Any, Any]:
        r"""
        Calculate `rel-coords` by `absolute-coords` and `relative-origin`.

        Parameters
        ----------
        origin_x, origin_y, abs_x, abs_y
            `absolute-coords` & `relative-origin`.
        angle
            Angle measured **anticlockwise** in degree from true north to the north of relative coordinate system.

        Returns
        -------
        tuple[Any, Any]
            `rel-coords`.

        Examples
        --------
        >>> from glatzel import geotool
        >>> geotool.CoordMigrate.rel(10, 20, 2, -1, 150.0)
        -3.5717967697244886 22.186533479473212
        """
        rel_x, rel_y = np.dot(
            cls.rotate_matrix(angle),
            np.stack([abs_x - origin_x, abs_y - origin_y]),
        )
        return rel_x, rel_y

    @classmethod
    def abs(cls, origin_x: float, origin_y: float, rel_x, rel_y, angle: float) -> tuple[Any, Any]:
        r"""
        Calculate `absolute-coords` by `relative-origin` and `relative-origin`.

        Parameters
        ----------
        origin_x, origin_y, rel_x, rel_y
            `relative-origin` & `rel-coords`.
        angle
            Angle measured **anticlockwise** in degree from `true north` to the `north of relative coordinate system`.

        Returns
        -------
        tuple[Any, Any]
            `absolute-coordss`.

        Examples
        --------
        >>> geotool.CoordMigrate.(10, 20, 2, -1, 150.0)
        8.767949192431123 21.866025403784437
        """
        abs_x, abs_y = np.dot(cls.rotate_matrix(-angle), np.stack([rel_x, rel_y])) + np.array(
            [origin_x, origin_y], np.float64
        )
        return abs_x, abs_y

    @classmethod
    def origin(
        cls,
        abs_x: float,
        abs_y: float,
        rel_x: float,
        rel_y: float,
        angle: float,
    ) -> tuple[float, float]:
        r"""
        Calculate `relative-origin` by `absolute-coords` and `rel-coords`.

        Parameters
        ----------
        abs_x, abs_y, rel_x, rel_y
            `absolute-coordss` & `relative coordinates`.
        angle
            Angle measured **anticlockwise** in degree from `true north` to the `north of relative coordinate system`.

        Returns
        -------
        tuple[float, float]
            `relative-origin`.

        Examples
        --------
        >>> from glatzel import geotool
        >>> geotool.CoordMigrate.abs(10, 20, 2, -1, 150.0)
        11.232050807568877 18.133974596215563
        """
        origin_x, origin_y = (
            np.array([abs_x, abs_y], np.float64) - np.dot(cls.rotate_matrix(-angle), [rel_x, rel_y])
        ).tolist()
        return origin_x, origin_y
print(CoordMigrate.abs(10, 20, 2, -1, 150.0))