/// Converts Cartesian coordinates (x, y, z) to cylindrical coordinates (r, u, z).
///
/// # Arguments
/// - `x`: The x-coordinate in Cartesian space (in any unit).
/// - `y`: The y-coordinate in Cartesian space (in any unit).
/// - `z`: The z-coordinate in Cartesian space (in any unit).
///
/// # Returns
/// A tuple `(r, u, z)` representing the cylindrical coordinates:
/// - `r`: The radial distance from the z-axis in the x-y plane.
/// - `u`: The azimuthal angle, the angle in the x-y plane (in radians).
/// - `z`: The same z-coordinate as input.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let (r, u, z) = geotool_algorithm::cartesian_to_cylindrical(1.2, 3.4, -5.6);
/// assert_approx_eq!(f64, r, 3.60555127546399, epsilon = 1e-6);
/// assert_approx_eq!(f64, u, 1.23150371234085, epsilon = 1e-6);
/// assert_approx_eq!(f64, z, -5.60000000000000, epsilon = 1e-6);
/// ```
pub fn cartesian_to_cylindrical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    ((x.powf(2.0) + y.powf(2.0)).sqrt(), y.atan2(x), z)
}

/// Converts Cartesian coordinates (x, y, z) to spherical coordinates (u, v, r).
///
/// # Arguments
/// - `x`: The x-coordinate in Cartesian space (in any unit).
/// - `y`: The y-coordinate in Cartesian space (in any unit).
/// - `z`: The z-coordinate in Cartesian space (in any unit).
///
/// # Returns
/// A tuple `(u, v, r)` representing the spherical coordinates:
/// - `u`: The azimuthal angle, the angle in the x-y plane (in radians).
/// - `v`: The polar angle, the angle from the z-axis (in radians).
/// - `r`: The radial distance from the origin.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let (u, v, r) = geotool_algorithm::cartesian_to_spherical(1.2, 3.4, -5.6);
/// assert_approx_eq!(f64, u, 1.23150371234085, epsilon = 1e-6);
/// assert_approx_eq!(f64, v, 2.5695540653144073, epsilon = 1e-6);
/// assert_approx_eq!(f64, r, 6.66033032213868, epsilon = 1e-6);
/// ```
pub fn cartesian_to_spherical(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = (x.powf(2.0) + y.powf(2.0) + z.powf(2.0)).sqrt();
    (y.atan2(x), (z / r).acos(), r)
}

/// Converts cylindrical coordinates (r, u, z) to Cartesian coordinates (x, y, z).
///
/// # Arguments
/// - `r`: The radial distance from the z-axis in the x-y plane.
/// - `u`: The azimuthal angle in the x-y plane (in radians).
/// - `z`: The z-coordinate (same as in cylindrical and Cartesian systems).
///
/// # Returns
/// A tuple `(x, y, z)` representing the Cartesian coordinates:
/// - `x`: The x-coordinate in Cartesian space.
/// - `y`: The y-coordinate in Cartesian space.
/// - `z`: The same z-coordinate as input.
///
/// # Example
/// ```
///  use float_cmp::assert_approx_eq;
/// let (x, y, z) = geotool_algorithm::cylindrical_to_cartesian(3.60555127546399, 1.23150371234085, -5.60000000000000);
/// assert_approx_eq!(f64, x, 1.2,epsilon = 1e-6);
/// assert_approx_eq!(f64, y, 3.4, epsilon = 1e-6);
/// assert_approx_eq!(f64, z, -5.6,epsilon = 1e-6);
/// ```
pub fn cylindrical_to_cartesian(r: f64, u: f64, z: f64) -> (f64, f64, f64) {
    (r * u.cos(), r * u.sin(), z)
}

/// Converts cylindrical coordinates (r, u, z) to spherical coordinates (u, v, r).
///
/// # Arguments
/// - `r`: The radial distance in the x-y plane.
/// - `u`: The azimuthal angle in the x-y plane (in radians).
/// - `z`: The z-coordinate (same as in cylindrical and spherical systems).
///
/// # Returns
/// A tuple `(u, v, r)` representing the spherical coordinates:
/// - `u`: The azimuthal angle in the x-y plane (same as cylindrical).
/// - `v`: The polar angle from the z-axis (in radians).
/// - `r`: The radial distance in the x-z plane (including z).
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let (u, v, r) = geotool_algorithm::cylindrical_to_spherical(3.60555127546399, 1.23150371234085, -5.60000000000000);
/// assert_approx_eq!(f64, u, 1.23150371234085, epsilon = 1e-6);
/// assert_approx_eq!(f64, v, 2.5695540653144073, epsilon = 1e-6);
/// assert_approx_eq!(f64, r, 6.66033032213868, epsilon = 1e-6);
/// ```
pub fn cylindrical_to_spherical(r: f64, u: f64, z: f64) -> (f64, f64, f64) {
    (u, r.atan2(z), (r.powf(2.0) + z.powf(2.0)).sqrt())
}

/// Converts spherical coordinates (u, v, r) to Cartesian coordinates (x, y, z).
///
/// # Arguments
/// - `u`: The azimuthal angle in the x-y plane (in radians).
/// - `v`: The polar angle from the z-axis (in radians).
/// - `r`: The radial distance from the origin.
///
/// # Returns
/// A tuple `(x, y, z)` representing the Cartesian coordinates:
/// - `x`: The x-coordinate in Cartesian space.
/// - `y`: The y-coordinate in Cartesian space.
/// - `z`: The z-coordinate in Cartesian space.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let (x, y, z) = geotool_algorithm::spherical_to_cartesian(1.23150371234085, 2.5695540653144073, 6.66033032213868);
/// assert_approx_eq!(f64, x, 1.2,epsilon = 1e-6);
/// assert_approx_eq!(f64, y, 3.4, epsilon = 1e-6);
/// assert_approx_eq!(f64, z, -5.6,epsilon = 1e-6);
/// ```
pub fn spherical_to_cartesian(u: f64, v: f64, r: f64) -> (f64, f64, f64) {
    (r * v.sin() * u.cos(), r * v.sin() * u.sin(), r * v.cos())
}

/// Converts spherical coordinates (u, v, r) to cylindrical coordinates (r, u, z).
///
/// # Arguments
/// - `u`: The azimuthal angle in the x-y plane (in radians).
/// - `v`: The polar angle from the z-axis (in radians).
/// - `r`: The radial distance from the origin.
///
/// # Returns
/// A tuple `(r, u, z)` representing the cylindrical coordinates:
/// - `r`: The radial distance from the z-axis in the x-y plane.
/// - `u`: The azimuthal angle in the x-y plane (same as spherical).
/// - `z`: The z-coordinate (same as spherical).
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let (r, u, z) = geotool_algorithm::spherical_to_cylindrical(1.23150371234085, 2.5695540653144073, 6.66033032213868);
/// assert_approx_eq!(f64, r, 3.60555127546399,epsilon = 1e-6);
/// assert_approx_eq!(f64, u, 1.23150371234085, epsilon = 1e-6);
/// assert_approx_eq!(f64, z, -5.60000000000000,epsilon = 1e-6);
/// ```
pub fn spherical_to_cylindrical(u: f64, v: f64, r: f64) -> (f64, f64, f64) {
    (r * v.sin(), u, r * v.cos())
}
