/// # References
/// - https://github.com/googollee/eviltransform/blob/master/rust/src/lib.rs
/// - https://github.com/billtian/wgtochina_lb-php/tree/master
/// - https://github.com/Leask/EvilTransform
use std::f64::consts::PI;
pub enum CryptoSpace {
    WGS84,
    GCJ02,
    BD09,
}

const EARTH_R: f64 = 6378137.0;
const _X_PI: f64 = PI * 3000.0 / 180.0;
const EE: f64 = 0.006_693_421_622_965_943;
fn out_of_china(lon: f64, lat: f64) -> bool {
    if !(72.004..=137.8347).contains(&lon) {
        return true;
    }
    if !(0.8293..=55.8271).contains(&lat) {
        return true;
    }
    false
}
fn transform(x: f64, y: f64) -> (f64, f64) {
    let xy = x * y;
    let abs_x = x.abs().sqrt();
    let x_pi = x * PI;
    let y_pi = y * PI;
    let d = 20.0 * (6.0 * x_pi).sin() + 20.0 * (2.0 * x_pi).sin();

    let mut lat = d;
    let mut lon = d;

    lat += 20.0 * (y_pi).sin() + 40.0 * (y_pi / 3.0).sin();
    lon += 20.0 * (x_pi).sin() + 40.0 * (x_pi / 3.0).sin();

    lat += 160.0 * (y_pi / 12.0).sin() + 320.0 * (y_pi / 30.0).sin();
    lon += 150.0 * (x_pi / 12.0).sin() + 300.0 * (x_pi / 30.0).sin();

    lat *= 2.0 / 3.0;
    lon *= 2.0 / 3.0;

    lat += -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * xy + 0.2 * abs_x;
    lon += 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * xy + 0.1 * abs_x;

    (lon, lat)
}
fn delta(lon: f64, lat: f64) -> (f64, f64) {
    let (d_lon, d_lat) = transform(lon - 105.0, lat - 35.0);
    let mut d_lat = d_lat;
    let mut d_lon = d_lon;
    let rad_lat = lat / 180.0 * PI;
    let mut magic = (rad_lat).sin();
    magic = 1.0 - EE * magic * magic;
    let sqrt_magic = (magic).sqrt();
    d_lat = (d_lat * 180.0) / ((EARTH_R * (1.0 - EE)) / (magic * sqrt_magic) * PI);
    d_lon = (d_lon * 180.0) / (EARTH_R / sqrt_magic * (rad_lat).cos() * PI);
    (d_lon, d_lat)
}
/// Converts coordinates from `BD09` to `GCJ02` coordinate system.
///
/// # Arguments
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
/// assert_approx_eq!(f64, p.0, 121.09626892329175, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, 30.608594105135296, epsilon = 1e-6);
/// ```
pub fn bd09_to_gcj02(bd09_lon: f64, bd09_lat: f64) -> (f64, f64) {
    let x = bd09_lon - 0.0065;
    let y = bd09_lat - 0.006;
    let z = (x * x + y * y).sqrt() - 0.00002 * (y * _X_PI).sin();
    let theta = y.atan2(x) - 0.000003 * (x * _X_PI).cos();
    let gcj02_lon = z * theta.cos();
    let gcj02_lat = z * theta.sin();
    (gcj02_lon, gcj02_lat)
}
/// Converts coordinates from `GCJ02` to `WGS84` coordinate system.
///
/// # Arguments
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
/// let p = (121.09626935575027, 30.608604331756705);
/// let p = geotool_algorithm::gcj02_to_wgs84(p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.0917077 , epsilon = 1e-5);
/// assert_approx_eq!(f64, p.1, 30.6107779 , epsilon = 1e-5);
/// ```
pub fn gcj02_to_wgs84(lon: f64, lat: f64) -> (f64, f64) {
    if out_of_china(lon, lat) {
        return (lon, lat);
    }
    let (d_lon, d_lat) = delta(lon, lat);
    (lon - d_lon, lat - d_lat)
}
/// Converts coordinates from `BD09` to `WGS84` coordinate system.
///
/// # Arguments
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
/// assert_approx_eq!(f64, p.0, 121.09170577473259, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, 30.610767662599578, epsilon = 1e-6);
/// ```
pub fn bd09_to_wgs84(lon: f64, lat: f64) -> (f64, f64) {
    let (gcj_lon, gcj_lat) = bd09_to_gcj02(lon, lat);
    gcj02_to_wgs84(gcj_lon, gcj_lat)
}
/// Converts coordinates from `GCJ02` to `BD09` coordinate system.
///
/// # Arguments
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
/// # Arguments
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
/// let p = (121.0917077,30.6107779 );
/// let p = geotool_algorithm::wgs84_to_gcj02(p.0, p.1);
/// println!("{},{}",p.0,p.1);
/// assert_approx_eq!(f64, p.0, 121.09626935575027, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, 30.608604331756705, epsilon = 1e-6);
/// ```
pub fn wgs84_to_gcj02(lon: f64, lat: f64) -> (f64, f64) {
    if out_of_china(lon, lat) {
        return (lon, lat);
    }
    let (d_lon, d_lat) = delta(lon, lat);
    (lon + d_lon, lat + d_lat)
}
/// Converts coordinates from `BD09` to `WGS84` coordinate system.
///
/// # Arguments
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
/// assert_approx_eq!(f64, p.0, 121.10271691314193, epsilon = 1e-5);
/// assert_approx_eq!(f64, p.1, 30.614836298418275, epsilon = 1e-5);
/// ```
pub fn wgs84_to_bd09(lon: f64, lat: f64) -> (f64, f64) {
    let (gcj_lon, gcj_lat) = wgs84_to_gcj02(lon, lat);
    gcj02_to_bd09(gcj_lon, gcj_lat)
}

// gcj2wgs_exact convert GCJ-02 coordinate(gcj_lat, gcj_lon) to WGS-84 coordinate.
///
/// To output WGS-84 coordinate's accuracy is less than 0.5m, set `threshold = 1e-6` and `max_iter = 30`.
///
/// # Arguments
///
/// - `gcj_lon`: Longitude in `GCJ02` coordinate system.
/// - `gcj_lat`: Latitude in `GCJ02` coordinate system.
/// - `threshold`: Error threshold.
/// - `max_iter``: Max iterations.
pub fn gcj02_to_wgs84_exact(
    gcj_lon: f64,
    gcj_lat: f64,
    threshold: f64,
    max_iter: usize,
) -> (f64, f64) {
    let (mut wgs_lon, mut wgs_lat) = gcj02_to_wgs84(gcj_lon, gcj_lat);

    let mut d_lon = (wgs_lon - gcj_lon).abs();
    let mut d_lat = (wgs_lat - gcj_lat).abs();

    let mut m_lon = wgs_lon - d_lon;
    let mut m_lat = wgs_lat - d_lat;
    let mut p_lon = wgs_lon + d_lon;
    let mut p_lat = wgs_lat + d_lat;

    for i in 0..max_iter {
        (wgs_lon, wgs_lat) = ((m_lon + p_lon) / 2.0, (m_lat + p_lat) / 2.0);
        let (tmp_lon, tmp_lat) = wgs84_to_gcj02(wgs_lon, wgs_lat);
        d_lon = tmp_lon - gcj_lon;
        d_lat = tmp_lat - gcj_lat;

        #[cfg(debug_assertions)]
        {
            tracing::debug!("step: {i}");
            tracing::debug!("wgs_lon: {wgs_lon}, wgs_lat: {wgs_lat}");
            tracing::debug!("d_lon: {d_lon:.6e}, d_lat: {d_lat:.6e}");
            tracing::debug!("p_lon: {p_lon}, p_lat: {p_lat}");
            tracing::debug!("m_lon: {m_lon}, m_lat: {m_lat}");
        }

        if d_lat.abs() < threshold && d_lon.abs() < threshold {
            return (wgs_lon, wgs_lat);
        }
        if d_lon > 0.0 {
            p_lon = wgs_lon;
        } else {
            m_lon = wgs_lon;
        }
        if d_lat > 0.0 {
            p_lat = wgs_lat;
        } else {
            m_lat = wgs_lat;
        }
    }

    ((m_lon + p_lon) / 2.0, (m_lat + p_lat) / 2.0)
}

// distance calculate the distance between point(lat_a, lon_a) and point(lat_b, lon_b), unit in meter.
pub fn distance_geo(lon_a: f64, lat_a: f64, lon_b: f64, lat_b: f64) -> f64 {
    let arc_lat_a = lat_a * PI / 180.0;
    let arc_lat_b = lat_b * PI / 180.0;
    let x = (arc_lat_a).cos() * (arc_lat_b).cos() * ((lon_a - lon_b) * PI / 180.0).cos();
    let y = (arc_lat_a).sin() * (arc_lat_b).sin();
    let s = (x + y).clamp(-1.0, 1.0);
    let alpha = s.acos();
    alpha * EARTH_R
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    #[test]
    fn test_gcj2wgs_exact() {
        tracing_subscriber::registry()
            .with(log_template::terminal_layer(LevelFilter::TRACE))
            .init();
        let (wgs_lon, wgs_lat) =
            super::gcj02_to_wgs84_exact(121.09626935575027, 30.608604331756705, 1e-6, 30);
        println!("{wgs_lon},{wgs_lat}");
        let d = super::distance_geo(wgs_lon, wgs_lat, 121.0917077, 30.6107779);
        println!(
            "delta_lon:{:.6e}, delta_lat:{:.6e}, delta_distance: {d}",
            (wgs_lon - 121.0917077).abs(),
            (wgs_lat - 30.6107779).abs()
        );
        assert!(d < 0.5)
    }
}
