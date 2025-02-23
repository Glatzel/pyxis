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
pub enum CryptoThresholdMode {
    Distance,
    LonLat,
}

const EARTH_R: f64 = 6378137.0;
const X_PI: f64 = PI * 3000.0 / 180.0;
const EE: f64 = 0.006_693_421_622_965_943;

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
/// - `bd09_lon`: Longitude in `BD09` coordinate system.
/// - `bd09_lat`: Latitude in `BD09` coordinate system.
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
/// let p = (121.10271732371203, 30.61484572185035);
/// let p = geotool_algorithm::bd09_to_gcj02(p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.09626935575027, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, 30.608604331756705, epsilon = 1e-6);
/// ```
pub fn bd09_to_gcj02(bd09_lon: f64, bd09_lat: f64) -> (f64, f64) {
    let x = bd09_lon - 0.0065;
    let y = bd09_lat - 0.006;
    let z = (x * x + y * y).sqrt() - 0.00002 * (y * X_PI).sin();
    let theta = y.atan2(x) - 0.000003 * (x * X_PI).cos();
    let gcj02_lon = z * theta.cos();
    let gcj02_lat = z * theta.sin();
    (gcj02_lon, gcj02_lat)
}

/// Converts coordinates from `GCJ02` to `WGS84` coordinate system.
///
/// # Arguments
///
/// - `gcj02_lon`: Longitude in `GCJ02` coordinate system.
/// - `gcj02_lat`: Latitude in `GCJ02` coordinate system.
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
pub fn gcj02_to_wgs84(gcj02_lon: f64, gcj02_lat: f64) -> (f64, f64) {
    let (d_lon, d_lat) = delta(gcj02_lon, gcj02_lat);
    (gcj02_lon - d_lon, gcj02_lat - d_lat)
}

/// Converts coordinates from `BD09` to `WGS84` coordinate system.
///
/// # Arguments
///
/// - `bd09_lon`: Longitude in `BD09` coordinate system.
/// - `bd09_lat`: Latitude in `BD09` coordinate system.
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
pub fn bd09_to_wgs84(bd09_lon: f64, bd09_lat: f64) -> (f64, f64) {
    let (gcj_lon, gcj_lat) = bd09_to_gcj02(bd09_lon, bd09_lat);
    gcj02_to_wgs84(gcj_lon, gcj_lat)
}

/// Converts coordinates from `GCJ02` to `BD09` coordinate system.
///
/// # Arguments
///
/// - `gcj02_lon`: Longitude in `GCJ02` coordinate system.
/// - `gcj02_lat`: Latitude in `GCJ02` coordinate system.
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
/// let p = (121.09626935575027, 30.608604331756705);
/// let p = geotool_algorithm::gcj02_to_bd09(p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.10271732371203, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, 30.61484572185035, epsilon = 1e-17);
/// ```
pub fn gcj02_to_bd09(gcj02_lon: f64, gcj02_lat: f64) -> (f64, f64) {
    let z =
        (gcj02_lon * gcj02_lon + gcj02_lat * gcj02_lat).sqrt() + 0.00002 * (gcj02_lat * X_PI).sin();
    let theta = gcj02_lat.atan2(gcj02_lon) + 0.000003 * (gcj02_lon * X_PI).cos();
    let bd09_lon = z * (theta).cos() + 0.0065;
    let bd09_lat = z * (theta).sin() + 0.006;
    (bd09_lon, bd09_lat)
}

/// Converts coordinates from `WGS84` to `GCJ02` coordinate system.
///
/// # Arguments
///
/// - `wgs84_lon`: Longitude in `WGS84` coordinate system.
/// - `wgs84_lat`: Latitude in `WGS84` coordinate system.
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
/// assert_approx_eq!(f64, p.0, 121.09626935575027, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, 30.608604331756705, epsilon = 1e-17);
/// ```
pub fn wgs84_to_gcj02(wgs84_lon: f64, wgs84_lat: f64) -> (f64, f64) {
    let (d_lon, d_lat) = delta(wgs84_lon, wgs84_lat);
    (wgs84_lon + d_lon, wgs84_lat + d_lat)
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
/// - `wgs84_lon`: Longitude in the `WGS84` coordinate system.
/// - `wgs84_lat`: Latitude in the `WGS84` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// let p = (121.0917077,30.6107779);
/// let p = geotool_algorithm::wgs84_to_bd09(p.0, p.1);
/// assert_approx_eq!(f64, p.0, 121.10271732371203, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, 30.61484572185035,  epsilon = 1e-17);
/// ```
pub fn wgs84_to_bd09(wgs84_lon: f64, wgs84_lat: f64) -> (f64, f64) {
    let (gcj_lon, gcj_lat) = wgs84_to_gcj02(wgs84_lon, wgs84_lat);
    gcj02_to_bd09(gcj_lon, gcj_lat)
}

/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use tracing_subscriber::layer::SubscriberExt;
/// use tracing_subscriber::util::SubscriberInitExt;
/// use tracing_subscriber::filter::LevelFilter;
/// tracing_subscriber::registry()
///     .with(log_template::terminal_layer(LevelFilter::TRACE))
///     .init();
/// let p = (121.10271732371203, 30.61484572185035);
/// let p = geotool_algorithm::crypto_exact(
///             p.0,
///             p.1,
///             geotool_algorithm::bd09_to_wgs84,
///             geotool_algorithm::wgs84_to_bd09,
///             1e-17,
///             geotool_algorithm::CryptoThresholdMode::LonLat,
///             1000,
///         );
/// assert_approx_eq!(f64, p.0, 121.0917077, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, 30.6107779, epsilon = 1e-17);
/// ```
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use tracing_subscriber::layer::SubscriberExt;
/// use tracing_subscriber::util::SubscriberInitExt;
/// use tracing_subscriber::filter::LevelFilter;
/// tracing_subscriber::registry()
///     .with(log_template::terminal_layer(LevelFilter::TRACE))
///     .init();
/// let p = (121.10271732371203, 30.61484572185035);
/// let p = geotool_algorithm::crypto_exact(
///             p.0,
///             p.1,
///             geotool_algorithm::bd09_to_wgs84,
///             geotool_algorithm::wgs84_to_bd09,
///             1e-3,
///             geotool_algorithm::CryptoThresholdMode::Distance,
///             1000,
///         );
/// assert_approx_eq!(f64, p.0, 121.0917077, epsilon = 1e-8);
/// assert_approx_eq!(f64, p.1, 30.6107779, epsilon = 1e-8);
/// ```
pub fn crypto_exact(
    src_lon: f64,
    src_lat: f64,
    crypto_fn: impl Fn(f64, f64) -> (f64, f64),
    inv_crypto_fn: impl Fn(f64, f64) -> (f64, f64),
    threshold: f64,
    threshold_mode: CryptoThresholdMode,
    max_iter: usize,
) -> (f64, f64) {
    let (mut dst_lon, mut dst_lat) = crypto_fn(src_lon, src_lat);

    let mut d_lon = (dst_lon - src_lon).abs();
    let mut d_lat = (dst_lat - src_lat).abs();

    let mut m_lon = dst_lon - d_lon;
    let mut m_lat = dst_lat - d_lat;
    let mut p_lon = dst_lon + d_lon;
    let mut p_lat = dst_lat + d_lat;

    for _i in 0..max_iter {
        (dst_lon, dst_lat) = ((m_lon + p_lon) / 2.0, (m_lat + p_lat) / 2.0);
        let (tmp_lon, tmp_lat) = inv_crypto_fn(dst_lon, dst_lat);
        d_lon = tmp_lon - src_lon;
        d_lat = tmp_lat - src_lat;

        #[cfg(feature = "log")]
        {
            tracing::trace!("iteration: {_i}");
            tracing::trace!("dst_lon: {dst_lon}, dst_lat: {dst_lat}");
            tracing::trace!("d_lon: {d_lon:.2e}, d_lat: {d_lat:.2e}");
            tracing::trace!("p_lon: {p_lon}, p_lat: {p_lat}");
            tracing::trace!("m_lon: {m_lon}, m_lat: {m_lat}");
            tracing::trace!(
                "distance: {}",
                haversine_distance(src_lon, src_lat, tmp_lon, tmp_lat)
            );
            if _i == max_iter - 1 {
                tracing::debug!("Exeed max iteration number: {max_iter}")
            };
        }

        match threshold_mode {
            CryptoThresholdMode::Distance
                if haversine_distance(src_lon, src_lat, tmp_lon, tmp_lat) < threshold =>
            {
                break;
            }
            CryptoThresholdMode::LonLat if d_lat.abs() < threshold && d_lon.abs() < threshold => {
                break;
            }
            _ => (),
        }
        if d_lat > 0.0 {
            p_lat = dst_lat;
        } else {
            m_lat = dst_lat;
        }
        if d_lon> 0.0 {
            p_lon = dst_lon;
        } else {
            m_lon = dst_lon;
        }

        // match (d_lon > 0.0, d_lat > 0.0, d_lon.abs() > d_lat.abs()) {
        //     (true, true, true) => {
        //         p_lon = dst_lon;
        //         p_lat = (p_lat + dst_lat) / 2.0;
        //     }
        //     (true, false, true) => {
        //         p_lon = dst_lon;
        //         m_lat = (m_lat + dst_lat) / 2.0;
        //     }
        //     (false, true, true) => {
        //         m_lon = dst_lon;
        //         p_lat = (p_lat + dst_lat) / 2.0;
        //     }
        //     (false, false, true) => {
        //         m_lon = dst_lon;
        //         m_lat = (m_lat + dst_lat) / 2.0;
        //     }
        //     (true, true, false) => {
        //         p_lon = (dst_lon + p_lon) / 2.0;
        //         p_lat = dst_lat;
        //     }
        //     (false, true, false) => {
        //         m_lon = (dst_lon + m_lon) / 2.0;
        //         p_lat = dst_lat
        //     }
        //     (true, false, false) => {
        //         p_lon = (dst_lon + p_lon) / 2.0;
        //         m_lat = dst_lat;
        //     }
        //     (false, false, false) => {
        //         m_lon = (dst_lon + m_lon) / 2.0;
        //         m_lat = dst_lat;
        //     }
        // }
    }

    (dst_lon, dst_lat)
}
/// distance calculate the distance between point(lat_a, lon_a) and point(lat_b, lon_b), unit in meter.
pub fn haversine_distance(lon_a: f64, lat_a: f64, lon_b: f64, lat_b: f64) -> f64 {
    let lat1_rad = lat_a.to_radians();
    let lon1_rad = lon_a.to_radians();
    let lat2_rad = lat_b.to_radians();
    let lon2_rad = lon_b.to_radians();

    let delta_lat = lat2_rad - lat1_rad;
    let delta_lon = lon2_rad - lon1_rad;

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_R * c
}
#[cfg(test)]
mod test {

    use rand::prelude::*;
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    use super::*;

    #[test]
    fn test_exact() {
        tracing_subscriber::registry()
            .with(log_template::terminal_layer(LevelFilter::ERROR))
            .init();
        let mut rng = rand::rng();
        for _ in 0..10000 {
            let wgs = (
                rng.random_range(72.004..137.8347),
                rng.random_range(0.8293..55.8271),
            );
            let gcj = wgs84_to_gcj02(wgs.0, wgs.1);
            let bd = wgs84_to_bd09(wgs.0, wgs.1);
            {
                let test_gcj = crypto_exact(
                    bd.0,
                    bd.1,
                    bd09_to_gcj02,
                    gcj02_to_bd09,
                    1e-20,
                    CryptoThresholdMode::LonLat,
                    1000,
                );
                if (test_gcj.0 - gcj.0).abs() > 1e-7 || (test_gcj.1 - gcj.1).abs() > 1e-7 {
                    println!(
                        "gcj,{},{},{},{},{},{:.2e},{:.2e}",
                        test_gcj.0,
                        test_gcj.1,
                        gcj.0,
                        gcj.1,
                        haversine_distance(test_gcj.0, test_gcj.1, gcj.0, gcj.1),
                        test_gcj.0 - gcj.0,
                        test_gcj.1 - gcj.1
                    )
                };
            }
            {
                let test_wgs = crypto_exact(
                    bd.0,
                    bd.1,
                    bd09_to_wgs84,
                    wgs84_to_bd09,
                    1e-20,
                    CryptoThresholdMode::LonLat,
                    1000,
                );
                if (test_wgs.0 - wgs.0).abs() > 1e-7 || (test_wgs.1 - wgs.1).abs() > 1e-7 {
                    println!(
                        "wgs,{},{},{},{},{},{:.2e},{:.2e}",
                        test_wgs.0,
                        test_wgs.1,
                        wgs.0,
                        wgs.1,
                        haversine_distance(test_wgs.0, test_wgs.1, wgs.0, wgs.1),
                        test_wgs.0 - wgs.0,
                        test_wgs.1 - wgs.1,
                    )
                };
            }
            {
                let test_wgs = crypto_exact(
                    gcj.0,
                    gcj.1,
                    gcj02_to_wgs84,
                    wgs84_to_gcj02,
                    1e-20,
                    CryptoThresholdMode::LonLat,
                    1000,
                );
                if (test_wgs.0 - wgs.0).abs() > 1e-7 || (test_wgs.1 - wgs.1).abs() > 1e-7 {
                    println!(
                        "wgs,{},{},{},{},{},{:.2e},{:.2e}",
                        test_wgs.0,
                        test_wgs.1,
                        wgs.0,
                        wgs.1,
                        haversine_distance(test_wgs.0, test_wgs.1, wgs.0, wgs.1),
                        test_wgs.0 - wgs.0,
                        test_wgs.1 - wgs.1,
                    )
                };
            }
        }
    }
}
