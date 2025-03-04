import numpy as np
from scipy.optimize import least_squares


def _transform_operation(T_x, T_y, T_z, R_x, R_y, R_z, s, source_points, target_points):
    rotation_matrix = np.array([[1, -R_z, R_y], [R_z, 1, -R_x], [-R_y, R_x, 1]])
    scale = 1 + s
    transformed_points = scale * (source_points @ rotation_matrix.T) + [T_x, T_y, T_z]
    residuals = transformed_points - target_points
    return residuals.flatten()


def transformation_residuals3(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_guess: tuple[float, float, float] = (0, 0, 0),
) -> tuple[float, float, float]:
    def transform_wrapper(params, source_points, target_points):
        T_x, T_y, T_z = params
        return _transform_operation(T_x, T_y, T_z, 0.0, 0.0, 0.0, 0.0, source_points, target_points)

    # Solve for parameters
    result = least_squares(transform_wrapper, initial_guess, args=(source_points, target_points))
    parameters = result.x
    return parameters


def transformation_residuals6(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_guess: tuple[float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0),
) -> tuple[float, float, float, float, float, float]:
    def transform_wrapper(params, source_points, target_points):
        T_x, T_y, T_z, R_x, R_y, R_z = params
        return _transform_operation(T_x, T_y, T_z, R_x, R_y, R_z, 0.0, source_points, target_points)

    # Solve for parameters
    result = least_squares(transform_wrapper, initial_guess, args=(source_points, target_points))
    parameters = result.x
    return parameters


def transformation_residuals7(
    source_points: np.ndarray,
    target_points: np.ndarray,
    initial_guess: tuple[float, float, float, float, float, float, float] = (0, 0, 0, 0, 0, 0, 0),
) -> tuple[float, float, float, float, float, float, float]:
    def transform_wrapper(params, source_points, target_points):
        T_x, T_y, T_z, R_x, R_y, R_z, s = params
        return _transform_operation(T_x, T_y, T_z, R_x, R_y, R_z, s, source_points, target_points)

    # Solve for parameters
    result = least_squares(transform_wrapper, initial_guess, args=(source_points, target_points))
    parameters = result.x
    return parameters
