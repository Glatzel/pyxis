use std::f64::consts::PI;
pub enum CryptoSpace {
    WGS84,
    GCJ02,
    BD09,
}

const _X_PI: f64 = PI * 3000.0 / 180.0;
const _A: f64 = 6378245.0;
const _EE: f64 = 0.006_693_421_622_965_943;
fn transform_lon(lon: f64, lat: f64) -> f64 {
    let mut ret =
        300.0 + lon + 2.0 * lat + 0.1 * lon * lon + 0.1 * lon * lat + 0.1 * lon.abs().sqrt();
    ret += (20.0 * (6.0 * lon * PI).sin() + 20.0 * (2.0 * lon * PI).sin()) * 2.0 / 3.0;
    ret += (20.0 * (lon * PI).sin() + 40.0 * (lon / 3.0 * PI).sin()) * 2.0 / 3.0;
    ret += (150.0 * (lon / 12.0 * PI).sin() + 300.0 * (lon / 30.0 * PI).sin()) * 2.0 / 3.0;
    ret
}
fn transform_lat(lon: f64, lat: f64) -> f64 {
    let mut ret =
        -100.0 + 2.0 * lon + 3.0 * lat + 0.2 * lat * lat + 0.1 * lon * lat + 0.2 * lon.abs().sqrt();
    ret += (20.0 * (6.0 * lon * PI).sin() + 20.0 * (2.0 * lon * PI).sin()) * 2.0 / 3.0;
    ret += (20.0 * (lat * PI).sin() + 40.0 * (lat / 3.0 * PI).sin()) * 2.0 / 3.0;
    ret += (160.0 * (lat / 12.0 * PI).sin() + 320.0 * (lat * PI / 30.0).sin()) * 2.0 / 3.0;
    ret
}
/// Converts coordinates from `BD09` to `GCJ02` coordinate system.
///
/// # Parameters
///
/// - `lon`: Longitude in `BD09` coordinate system.
/// - `lat`: Latitude in `BD09` coordinate system.
///
/// # Returns
///
/// A tuple `(lon, lat)` representing the coordinates in the `GCJ02` coordinate system:
/// - `lon`: Longitude in the `GCJ02` coordinate system.
/// - `lat`: Latitude in the `GCJ02` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let p = (121.10271691314193, 30.614836298418275);
/// let p = geotool_algorithm::bd09_to_gcj02(p.0, p.1);
/// eprintln!("{},{}", p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.09626892329175, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, 30.608594105135296, epsilon = 1e-6);
/// ```
pub fn bd09_to_gcj02(lon: f64, lat: f64) -> (f64, f64) {
    let x = lon - 0.0065;
    let y = lat - 0.006;
    let z = (x * x + y * y).sqrt() - 0.00002 * (y * _X_PI).sin();
    let theta = y.atan2(x) - 0.000003 * (x * _X_PI).cos();
    let dest_lon = z * theta.cos();
    let dest_lat = z * theta.sin();
    (dest_lon, dest_lat)
}
/// Converts coordinates from `GCJ02` to `WGS84` coordinate system.
///
/// # Parameters
///
/// - `lon`: Longitude in `GCJ02` coordinate system.
/// - `lat`: Latitude in `GCJ02` coordinate system.
///
/// # Returns
///
/// A tuple `(lon, lat)` representing the coordinates in the `WGS84` coordinate system:
/// - `lon`: Longitude in the `WGS84` coordinate system.
/// - `lat`: Latitude in the `WGS84` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let p = (121.09626892329175, 30.608594105135296);
/// let p = geotool_algorithm::gcj02_to_wgs84(p.0, p.1);
/// eprintln!("{},{}", p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.09170577473259, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, 30.610767662599578, epsilon = 1e-6);
/// ```
pub fn gcj02_to_wgs84(lon: f64, lat: f64) -> (f64, f64) {
    let dlat = transform_lat(lon - 105.0, lat - 35.0);
    let dlon = transform_lon(lon - 105.0, lat - 35.0);
    let radlat = lat / 180.0 * PI;
    let magic = radlat.sin();
    let magic = 1.0 - _EE * magic * magic;
    let sqrtmagic = magic.sqrt();
    let dlat = (dlat * 180.0) / ((_A * (1.0 - _EE)) / (magic * sqrtmagic) * PI);
    let dlon = (dlon * 180.0) / (_A / sqrtmagic * radlat.cos() * PI);
    let mglat = lat + dlat;
    let mglon = lon + dlon;
    (lon * 2.0 - mglon, lat * 2.0 - mglat)
}
/// Converts coordinates from `BD09` to `WGS84` coordinate system.
///
/// # Parameters
///
/// - `lon`: Longitude in `BD09` coordinate system.
/// - `lat`: Latitude in `BD09` coordinate system.
///
/// # Returns
///
/// A tuple `(lon, lat)` representing the coordinates in the `WGS84` coordinate system:
/// - `lon`: Longitude in the `WGS84` coordinate system.
/// - `lat`: Latitude in the `WGS84` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let p = (121.10271691314193, 30.614836298418275);
/// let p = geotool_algorithm::bd09_to_wgs84(p.0, p.1);
/// eprintln!("{},{}", p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.09170577473259, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, 30.610767662599578, epsilon = 1e-6);
/// ```
pub fn bd09_to_wgs84(lon: f64, lat: f64) -> (f64, f64) {
    let (gcj_lon, gcj_lat) = bd09_to_gcj02(lon, lat);
    gcj02_to_wgs84(gcj_lon, gcj_lat)
}
/// Converts coordinates from `GCJ02` to `BD09` coordinate system.
///
/// # Parameters
///
/// - `lon`: Longitude in `GCJ02` coordinate system.
/// - `lat`: Latitude in `GCJ02` coordinate system.
///
/// # Returns
///
/// A tuple `(lon, lat)` representing the coordinates in the `BD09` coordinate system:
/// - `lon`: Longitude in the `BD09` coordinate system.
/// - `lat`: Latitude in the `BD09` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let p = (121.09626892329175, 30.608594105135296);
/// let p = geotool_algorithm::gcj02_to_bd09(p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.10271691314193, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, 30.614836298418275, epsilon = 1e-6);
/// ```
pub fn gcj02_to_bd09(lon: f64, lat: f64) -> (f64, f64) {
    let z = (lon * lon + lat * lat).sqrt() + 0.00002 * (lat * _X_PI).sin();
    let theta = lat.atan2(lon) + 0.000003 * (lon * _X_PI).cos();
    let dest_lon = z * (theta).cos() + 0.0065;
    let dest_lat = z * (theta).sin() + 0.006;
    (dest_lon, dest_lat)
}
/// Converts coordinates from `WGS84` to `GCJ02` coordinate system.
///
/// # Parameters
///
/// - `lon`: Longitude in `WGS84` coordinate system.
/// - `lat`: Latitude in `WGS84` coordinate system.
///
/// # Returns
///
/// A tuple `(lon, lat)` representing the coordinates in the `GCJ02` coordinate system:
/// - `lon`: Longitude in the `GCJ02` coordinate system.
/// - `lat`: Latitude in the `GCJ02` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let p = (121.09170577473259, 30.610767662599578);
/// let p = geotool_algorithm::wgs84_to_gcj02(p.0, p.1);
/// eprintln!("{},{}", p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.09626892329175, epsilon = 1e-5);
/// assert_approx_eq!(f64, p.1, 30.608594105135296, epsilon = 1e-5);
/// ```
pub fn wgs84_to_gcj02(lon: f64, lat: f64) -> (f64, f64) {
    let dlat = transform_lat(lon - 105.0, lat - 35.0);
    let dlon = transform_lon(lon - 105.0, lat - 35.0);
    let radlat = lat / 180.0 * PI;
    let magic = radlat.sin();
    let magic = 1.0 - _EE * magic * magic;
    let sqrt_magic = (magic).sqrt();
    let dlat = (dlat * 180.0) / ((_A * (1.0 - _EE)) / (magic * sqrt_magic) * PI);
    let dlon = (dlon * 180.0) / (_A / sqrt_magic * radlat.cos() * PI);
    let dest_lat = lat + dlat;
    let dest_lon = lon + dlon;
    (dest_lon, dest_lat)
}
/// Converts coordinates from `BD09` to `WGS84` coordinate system.
///
/// # Parameters
///
/// - `lon`: Longitude in `BD09` coordinate system.
/// - `lat`: Latitude in `BD09` coordinate system.
///
/// # Returns
///
/// A tuple `(lon, lat)` representing the coordinates in the `WGS84` coordinate system:
/// - `lon`: Longitude in the `WGS84` coordinate system.
/// - `lat`: Latitude in the `WGS84` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let p = (121.09170577473259, 30.610767662599578);
/// let p = geotool_algorithm::wgs84_to_bd09(p.0, p.1);
/// eprintln!("{},{}", p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.10271691314193, epsilon = 1e-5);
/// assert_approx_eq!(f64, p.1, 30.614836298418275, epsilon = 1e-5);
/// ```
pub fn wgs84_to_bd09(lon: f64, lat: f64) -> (f64, f64) {
    let (gcj_lon, gcj_lat) = wgs84_to_gcj02(lon, lat);
    gcj02_to_bd09(gcj_lon, gcj_lat)
}
