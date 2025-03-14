use crate::GeoFloat;
/// Converts polar coordinates (r, u) to Cartesian coordinates (x, y).
///
/// # Arguments
/// - `r`: The radius.
/// - `u`: The azimuthal angle.
///
/// # Returns
/// A tuple `(x, y)` representing the Cartesian coordinates:
/// - `x`: The x-coordinate in Cartesian space.
/// - `y`: The y-coordinate in Cartesian space.
///
/// # Example
/// ```
///  use float_cmp::assert_approx_eq;
/// let (x, y) = pyxis::polar_to_cartesian(3.605551275463989, 1.2315037123408519);
/// assert_approx_eq!(f64, x, 1.2,epsilon = 1e-15);
/// assert_approx_eq!(f64, y, 3.4, epsilon = 1e-15);
/// ```
pub fn polar_to_cartesian<T>(r: T, theta: T) -> (T, T)
where
    T: GeoFloat,
{
    (r * theta.cos(), r * theta.sin())
}
/// Converts Cartesian coordinates (x, y) to polar coordinates (r, theta).
///
/// # Arguments
/// - `x`: The x-coordinate in Cartesian space (in any unit).
/// - `y`: The y-coordinate in Cartesian space (in any unit).
///
/// # Returns
/// A tuple `(r, theta)` representing the cylindrical coordinates:
/// - `r`: The radius.
/// - `theta`: The azimuthal angle.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let (r, u) = pyxis::cartesian_to_polar(1.2, 3.4);
/// assert_approx_eq!(f64, r, 3.605551275463989, epsilon = 1e-17);
/// assert_approx_eq!(f64, u, 1.2315037123408519, epsilon = 1e-17);
/// ```
pub fn cartesian_to_polar<T>(x: T, y: T) -> (T, T)
where
    T: GeoFloat,
{
    ((x.powi(2) + y.powi(2)).sqrt(), y.atan2(x))
}

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
/// let (r, u, z) = pyxis::cartesian_to_cylindrical(1.2, 3.4, -5.6);
/// assert_approx_eq!(f64, r, 3.605551275463989, epsilon = 1e-17);
/// assert_approx_eq!(f64, u, 1.2315037123408519, epsilon = 1e-17);
/// assert_approx_eq!(f64, z, -5.60000000000000, epsilon = 1e-17);
/// ```
pub fn cartesian_to_cylindrical<T>(x: T, y: T, z: T) -> (T, T, T)
where
    T: GeoFloat,
{
    ((x.powi(2) + y.powi(2)).sqrt(), y.atan2(x), z)
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
/// let (u, v, r) = pyxis::cartesian_to_spherical(1.2, 3.4, -5.6);
/// assert_approx_eq!(f64, u, 1.2315037123408519, epsilon = 1e-15);
/// assert_approx_eq!(f64, v, 2.5695540653144073, epsilon = 1e-15);
/// assert_approx_eq!(f64, r, 6.660330322138685, epsilon = 1e-15);
/// ```
pub fn cartesian_to_spherical<T>(x: T, y: T, z: T) -> (T, T, T)
where
    T: GeoFloat,
{
    let r = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();
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
/// let (x, y, z) = pyxis::cylindrical_to_cartesian(3.605551275463989, 1.2315037123408519, -5.60000000000000);
/// assert_approx_eq!(f64, x, 1.2,epsilon = 1e-15);
/// assert_approx_eq!(f64, y, 3.4, epsilon = 1e-15);
/// assert_approx_eq!(f64, z, -5.6,epsilon = 1e-15);
/// ```
pub fn cylindrical_to_cartesian<T>(r: T, u: T, z: T) -> (T, T, T)
where
    T: GeoFloat,
{
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
/// let (u, v, r) = pyxis::cylindrical_to_spherical(3.605551275463989, 1.2315037123408519, -5.60000000000000);
/// assert_approx_eq!(f64, u, 1.2315037123408519, epsilon = 1e-15);
/// assert_approx_eq!(f64, v, 2.5695540653144073, epsilon = 1e-15);
/// assert_approx_eq!(f64, r, 6.660330322138685, epsilon = 1e-15);
/// ```
pub fn cylindrical_to_spherical<T>(r: T, u: T, z: T) -> (T, T, T)
where
    T: GeoFloat,
{
    (u, r.atan2(z), (r.powi(2) + z.powi(2)).sqrt())
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
/// let (x, y, z) = pyxis::spherical_to_cartesian(1.2315037123408519, 2.5695540653144073, 6.660330322138685);
/// assert_approx_eq!(f64, x, 1.2,epsilon = 1e-15);
/// assert_approx_eq!(f64, y, 3.4, epsilon = 1e-15);
/// assert_approx_eq!(f64, z, -5.6,epsilon = 1e-15);
/// ```
pub fn spherical_to_cartesian<T>(u: T, v: T, r: T) -> (T, T, T)
where
    T: GeoFloat,
{
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
/// let (r, u, z) = pyxis::spherical_to_cylindrical(1.2315037123408519, 2.5695540653144073, 6.660330322138685);
/// assert_approx_eq!(f64, r, 3.605551275463989,epsilon = 1e-15);
/// assert_approx_eq!(f64, u, 1.2315037123408519, epsilon = 1e-15);
/// assert_approx_eq!(f64, z, -5.60000000000000,epsilon = 1e-15);
/// ```
pub fn spherical_to_cylindrical<T>(u: T, v: T, r: T) -> (T, T, T)
where
    T: GeoFloat,
{
    (r * v.sin(), u, r * v.cos())
}
