/// Converts projected XY coordinates from the height compensation plane to the sea level plane.
///
/// # Parameters
///
/// - `xc`, `yc`: Coordinates on the height compensation plane (in meters).
/// - `hb`: Elevation of the height compensation plane (in meters).
/// - `radius`: Radius of the Earth (in meters).
/// - `x0`, `y0`: Coordinate system origin (in meters).
///
/// # Returns
///
/// A tuple containing the projected XY coordinates of the sea level plane (in meters).
///
///
/// # References
/// - 杨元兴. (2008). 抵偿高程面的选择与计算. 城市勘测 (02), 72-74.
///
/// # Examples
/// ```
/// use float_cmp::approx_eq;
/// let p = (469704.6693, 2821940.796);
/// let p = geotool_algorithm::datum_compense(p.0, p.1, 400.0, 6378_137.0, 500_000.0, 0.0);
/// assert!(approx_eq!(f64, p.0, 469706.56912942, epsilon = 1e-6));
/// assert!(approx_eq!(f64, p.1, 2821763.83123231, epsilon = 1e-6));
/// ```
pub fn datum_compense(xc: f64, yc: f64, hb: f64, radius: f64, x0: f64, y0: f64) -> (f64, f64) {
    let ratio = hb / radius;
    let factor = ratio / (1.0 + ratio);
    let xc = xc - factor * (xc - x0);
    let yc = yc - factor * (yc - y0);
    (xc, yc)
}
/// Converts geodetic coordinates (longitude/L, latitude/B, height/H) to Cartesian coordinates (X, Y, Z).
///
/// # Parameters
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
/// use float_cmp::approx_eq;
/// use geotool_algorithm::Ellipsoid;
/// let ellipsoid = Ellipsoid::from(6378137.0, 298.257223563);
/// let (x, y, z) = geotool_algorithm::lbh2xyz(48.8566, 2.3522, 35.0, &ellipsoid);
/// println!("{},{},{}", x, y, z);
/// assert!(approx_eq!(f64, x, 4192979.6198897623, epsilon = 1e-6));
/// assert!(approx_eq!(f64, y, 4799159.563725418, epsilon = 1e-6));
/// assert!(approx_eq!(f64, z, 260022.66015989496, epsilon = 1e-6));
/// ```
pub fn lbh2xyz(lon: f64, lat: f64, height: f64, ellipsoid: &crate::Ellipsoid) -> (f64, f64, f64) {
    // Constants from the ellipsoid
    let a = ellipsoid.semi_major_axis(); // Semi-major axis
    let e2 = ellipsoid.eccentricity2(); // Squared eccentricity

    // Convert latitude and longitude from degrees to radians
    let lat_rad = lat.to_radians();
    let lon_rad = lon.to_radians();

    let n = a / (1.0 - e2 * lat_rad.sin().powi(2)).sqrt();
    let x = (n + height) * lat_rad.cos() * lon_rad.cos();
    let y = (n + height) * lat_rad.cos() * lon_rad.sin();
    let z = ((1.0 - e2) * n + height) * lat_rad.sin();
    (x, y, z)
}
/// Converts Cartesian coordinates (X, Y, Z) to geodetic coordinates (Longitude, Latitude, Height).
///
/// # Parameters
///
/// - `x`: Cartesian X-coordinate(s) in meters.
/// - `y`: Cartesian Y-coordinate(s) in meters.
/// - `z`: Cartesian Z-coordinate(s) in meters.
/// - `ellipsoid`: The ellipsoid parameters, which include the semi-major axis and inverse flattening.
///
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
/// use float_cmp::approx_eq;
/// use geotool_algorithm::Ellipsoid;
///
/// let ellipsoid = Ellipsoid::from(6378137.0, 298.257223563);
/// let (x, y, z) = geotool_algorithm::xyz2lbh(
///     4192979.6198897623,
///     4799159.563725418,
///     260022.66015989496,
///     &ellipsoid,
/// );
/// println!("{},{},{}", x, y, z);
/// assert!(approx_eq!(f64, x, 48.8566, epsilon = 1e-6));
/// assert!(approx_eq!(f64, y, 2.3522, epsilon = 1e-6));
/// assert!(approx_eq!(f64, z, 35.0, epsilon = 1e-6));
/// ```
pub fn xyz2lbh(
    x: f64,
    y: f64,
    z: f64,
    ellipsoid: &crate::Ellipsoid,

) -> (f64, f64, f64) {
    let tolerance = 1e-17;
    let max_iterations = 100;

    // Constants from the ellipsoid
    let a = ellipsoid.semi_major_axis(); // Semi-major axis
    let e2 = ellipsoid.eccentricity2(); // Squared eccentricity

    // Longitude
    let longitude = y.atan2(x);

    // Initial calculations
    let p = (x.powi(2) + y.powi(2)).sqrt(); // Projection on equatorial plane
    let mut latitude = z.atan2(p * (1.0 - e2)); // Initial latitude estimate
    let mut n = a / (1.0 - e2 * latitude.sin().powi(2)).sqrt(); // Radius of curvature
    let mut height = p / latitude.cos() - n;

    // Iterative refinement of latitude
    for _ in 0..max_iterations {
        let sin_lat = latitude.sin();
        n = a / (1.0 - e2 * sin_lat.powi(2)).sqrt();
        let new_latitude = z.atan2(p * (1.0 - e2 * n / (n + height)));
        height = p / new_latitude.cos() - n;
        if (new_latitude - latitude).abs() < tolerance {
            break;
        }
        latitude = new_latitude;
    }

    // Convert radians to degrees
    let longitude = longitude.to_degrees();
    let latitude = latitude.to_degrees();

    (longitude, latitude, height)
}
