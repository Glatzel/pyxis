use crate::GeoFloat;

/// Creates a 2D rotation matrix for a given angle in radians.
///
/// This function generates the 2x2 rotation matrix that can be used to rotate
/// points in 2D space by a given angle.
///
/// # Arguments
///
/// * `radians` - The angle to rotate the matrix by, in radians.
///
/// # Returns
///
/// A 2x2 rotation matrix as a 2D array:
/// [
///     [cos(angle), -sin(angle)],
///     [sin(angle), cos(angle)]
/// ]
///
/// # Example
///
/// ```rust
/// use float_cmp::assert_approx_eq;
/// let radians = 30.0f64.to_radians(); // 90 degrees in radians
/// let m = pyxis_algorithm::rotate_matrix_2d(radians);
/// assert_approx_eq!(f64, m[0][0], radians.cos(), epsilon = 1e-17);
/// assert_approx_eq!(f64, m[0][1], -radians.sin(), epsilon = 1e-17);
/// assert_approx_eq!(f64, m[1][0], radians.sin(), epsilon = 1e-17);
/// assert_approx_eq!(f64, m[1][1], radians.cos(), epsilon = 1e-17);
/// ```
pub fn rotate_matrix_2d<T>(radians: T) -> [[T; 2]; 2]
where
    T: GeoFloat,
{
    [
        [radians.cos(), -radians.sin()],
        [radians.sin(), radians.cos()],
    ]
}

/// # Example
///
/// ```rust
/// use float_cmp::assert_approx_eq;
/// let radians = 30.0f64.to_radians(); // 90 degrees in radians
/// let m = pyxis_algorithm::rotate_matrix_2d(radians);
/// let result=pyxis_algorithm::rotate_2d (3.0,2.0,&m);
/// assert_approx_eq!(f64, result.0, 3.0 * radians.cos() - 2.0 * radians.sin(), epsilon = 1e-17);
/// assert_approx_eq!(f64, result.1, 3.0 * radians.sin() + 2.0 * radians.cos(), epsilon = 1e-17);
/// ```
pub fn rotate_2d<T>(x: T, y: T, rotate_matrix: &[[T; 2]; 2]) -> (T, T)
where
    T: GeoFloat,
{
    let out_x = rotate_matrix[0][0] * x + rotate_matrix[0][1] * y;
    let out_y = rotate_matrix[1][0] * x + rotate_matrix[1][1] * y;
    (out_x, out_y)
}
