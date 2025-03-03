use crate::GeoFloat;

/// # Examples
/// ```
/// use float_cmp::assert_approx_eq;
/// let m=geotool_algorithm::rotate_matrix_2d(150.0f64.to_radians());
/// let result=geotool_algorithm::migrate::rel_2d(10.0, 20.0, 2.0, -1.0,&m);
/// assert_approx_eq!(f64, result.0, -3.5717967697244886, epsilon = 1e-17);
/// assert_approx_eq!(f64, result.1, 22.186533479473212, epsilon = 1e-18);
/// ```
pub fn rel_2d<T>(
    origin_x: T,
    origin_y: T,
    abs_x: T,
    abs_y: T,
    rotate_matrix: &[[T; 2]; 2],
) -> (T, T)
where
    T: GeoFloat,
{
    let temp_x = abs_x - origin_x;
    let temp_y = abs_y - origin_y;
    (
        rotate_matrix[0][0] * temp_x - rotate_matrix[0][1] * temp_y,
        -rotate_matrix[1][0] * temp_x + rotate_matrix[1][1] * temp_y,
    )
}
/// # Examples
/// ```
/// use float_cmp::assert_approx_eq;
/// let m=geotool_algorithm::rotate_matrix_2d(150.0f64.to_radians());
/// let result=geotool_algorithm::migrate::abs_2d(10.0, 20.0, 2.0, -1.0, &m);
/// assert_approx_eq!(f64, result.0, 8.767949192431123, epsilon = 1e-17);
/// assert_approx_eq!(f64, result.1, 21.866025403784437, epsilon = 1e-17);
/// ```
pub fn abs_2d<T>(
    origin_x: T,
    origin_y: T,
    rel_x: T,
    rel_y: T,
    rotate_matrix: &[[T; 2]; 2],
) -> (T, T)
where
    T: GeoFloat,
{
    (
        rotate_matrix[0][0] * rel_x + rotate_matrix[0][1] * rel_y + origin_x,
        rotate_matrix[1][0] * rel_x + rotate_matrix[1][1] * rel_y + origin_y,
    )
}
/// # Examples
/// ```
/// use float_cmp::assert_approx_eq;
/// let m=geotool_algorithm::rotate_matrix_2d(150.0f64.to_radians());
/// let result=geotool_algorithm::migrate::origin_2d(10.0, 20.0, 2.0, -1.0, &m);
/// assert_approx_eq!(f64, result.0, 11.232050807568877, epsilon = 1e-17);
/// assert_approx_eq!(f64, result.1, 18.133974596215563, epsilon = 1e-17);
/// ```
pub fn origin_2d<T>(abs_x: T, abs_y: T, rel_x: T, rel_y: T, rotate_matrix: &[[T; 2]; 2]) -> (T, T)
where
    T: GeoFloat,
{
    (
        -rotate_matrix[0][0] * rel_x - rotate_matrix[0][1] * rel_y + abs_x,
        -rotate_matrix[1][0] * rel_x - rotate_matrix[1][1] * rel_y + abs_y,
    )
}

// pub fn rotate_matrix_3d(rx: f64, ry: f64, rz: f64, order: RotareOrder) -> [[f64; 3]; 3] {
//     [
//         [radians.cos(), radians.sin()],
//         [-radians.sin(), radians.cos()],
//     ]
// }
