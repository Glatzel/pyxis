use crate::GeoFloat;

/// Converts geodetic coordinates (longitude/L, latitude/B, height/H) to Cartesian coordinates (X, Y, Z).
///
/// # Arguments
///
/// - `lon`: Geodetic longitude(s) in degrees. Can be a single value or an array of values.
/// - `lat`: Geodetic latitude(s) in degrees. Can be a single value or an array of values.
/// - `height`: Ellipsoidal height(s) in meters. Can be a single value or an array of values.
/// - `ellipsoid`: The ellipsoid parameters, which include the semi-major axis and inverse flattening.
///
/// # Returns
///
/// A tuple containing:
/// - `X`: Cartesian X-coordinate(s) in meters. Same shape as the input latitude, longitude, and height.
/// - `Y`: Cartesian Y-coordinate(s) in meters. Same shape as the input latitude, longitude, and height.
/// - `Z`: Cartesian Z-coordinate(s) in meters. Same shape as the input latitude, longitude, and height.
///
/// # Notes
/// - The conversion uses the ellipsoid's semi-major axis and inverse flattening.
/// - Latitude and longitude are provided in degrees and are internally converted to radians.
///
/// # Examples
///
/// Convert a single geodetic coordinate:
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::Ellipsoid;
/// let ellipsoid = Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257223563);
/// let (x, y, z) = pyxis::lbh2xyz(48.8566, 2.3522, 35.0, &ellipsoid);
/// println!("{},{},{}", x, y, z);
/// assert_approx_eq!(f64, x, 4192979.6198897623, epsilon = 1e-17);
/// assert_approx_eq!(f64, y, 4799159.563725418, epsilon = 1e-17);
/// assert_approx_eq!(f64, z, 260022.66015989496, epsilon = 1e-17);
/// ```
pub fn lbh2xyz<T>(lon: T, lat: T, height: T, ellipsoid: &crate::Ellipsoid<T>) -> (T, T, T)
where
    T: GeoFloat,
{
    // Constants from the ellipsoid
    let a = ellipsoid.semi_major_axis(); // Semi-major axis
    let e2 = ellipsoid.eccentricity2(); // Squared eccentricity

    // Convert latitude and longitude from degrees to radians
    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();

    let n = a / (T::ONE - e2 * lat_rad.sin().powi(2)).sqrt();
    let x = (n + height) * lat_rad.cos() * lon_rad.cos();
    let y = (n + height) * lat_rad.cos() * lon_rad.sin();
    let z = ((T::ONE - e2) * n + height) * lat_rad.sin();
    (x, y, z)
}
/// Converts Cartesian coordinates (X, Y, Z) to geodetic coordinates (Longitude, Latitude, Height).
///
/// # Arguments
///
/// - `x`: Cartesian X-coordinate(s) in meters.
/// - `y`: Cartesian Y-coordinate(s) in meters.
/// - `z`: Cartesian Z-coordinate(s) in meters.
/// - `ellipsoid`: The ellipsoid parameters, which include the semi-major axis and inverse flattening.
/// - `threshold`: Error threshold
/// - `max_iter`: Max iterations
/// # Returns
///
/// A tuple containing:
/// - `longitude`: Longitude in degrees.
/// - `latitude`: Latitude in degrees.
/// - `height`: Height above the reference ellipsoid in meters.
///
/// # Notes
/// - The function assumes the ellipsoid parameters are provided via the `ellipsoid` struct.
///
/// # Examples
/// Convert Cartesian coordinates to geodetic coordinates:
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::Ellipsoid;
///
/// let ellipsoid = Ellipsoid::from_semi_major_and_invf(6378137.0, 298.257223563);
/// let (x, y, z) = pyxis::xyz2lbh(
///     4192979.6198897623,
///     4799159.563725418,
///     260022.66015989496,
///     &ellipsoid,
///     1e-17,
///     25
/// );
/// println!("{},{},{}", x, y, z);
/// assert_approx_eq!(f64, x, 48.8566, epsilon = 1e-17);
/// assert_approx_eq!(f64, y, 2.3522, epsilon = 1e-7);
/// assert_approx_eq!(f64, z, 35.0, epsilon = 1e-17);
/// ```
pub fn xyz2lbh<T>(
    x: T,
    y: T,
    z: T,
    ellipsoid: &crate::Ellipsoid<T>,
    threshold: T,
    max_iter: usize,
) -> (T, T, T)
where
    T: GeoFloat,
{
    // Constants from the ellipsoid
    let a = ellipsoid.semi_major_axis(); // Semi-major axis
    let e2: T = ellipsoid.eccentricity2(); // Squared eccentricity

    // Longitude
    let longitude = y.atan2(x);

    // Initial calculations
    let p = (x.powi(2) + y.powi(2)).sqrt(); // Projection on equatorial plane
    let mut latitude = z.atan2(p * (T::ONE - e2)); // Initial latitude estimate
    let mut n: T = a / (T::ONE - e2 * latitude.sin().powi(2)).sqrt(); // Radius of curvature
    let mut height = p / latitude.cos() - n;

    // Iterative refinement of latitude
    for _i in 0..max_iter {
        let sin_lat = latitude.sin();
        n = a / (T::ONE - e2 * sin_lat.powi(2)).sqrt();
        let new_latitude = z.atan2(p * (T::ONE - e2 * n / (n + height)));
        height = p / new_latitude.cos() - n;

        clerk::trace!("iteration: {_i}");
        clerk::trace!("latitude: {}", latitude.to_degrees());
        clerk::trace!("new_latitude: {}", new_latitude.to_degrees());
        clerk::trace!(
            "delta: {}",
            (new_latitude.to_degrees() - latitude.to_degrees()).abs()
        );
        if _i == max_iter - 1 {
            clerk::debug!("Exceed max iteration number: {max_iter}");
        };

        if (new_latitude.to_degrees() - latitude.to_degrees()).abs() < threshold {
            break;
        }
        latitude = new_latitude;
    }

    // Convert radians to degrees
    let longitude = longitude.to_degrees();
    let latitude = latitude.to_degrees();

    (longitude, latitude, height)
}
