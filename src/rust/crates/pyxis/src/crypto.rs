/// # References
/// - https://github.com/googollee/eviltransform/blob/master/rust/src/lib.rs
/// - https://github.com/billtian/wgtochina_lb-php/tree/master
/// - https://github.com/Leask/EvilTransform
/// - https://github.com/wandergis/coordtransform
/// - https://blog.csdn.net/coolypf/article/details/8569813
/// - https://github.com/Artoria2e5/PRCoords/blob/master/js/PRCoords.js
use crate::types::{ConstEllipsoid, GeoFloat, num};

pub const WGS84_LON: f64 = 121.0917077;
pub const WGS84_LAT: f64 = 30.6107779;
pub const GCJ02_LON: f64 = 121.09626927850977;
pub const GCJ02_LAT: f64 = 30.608604368560773;
pub const BD09_LON: f64 = 121.10271724622564;
pub const BD09_LAT: f64 = 30.61484575976839;
pub enum CryptoSpace {
    WGS84,
    GCJ02,
    BD09,
}
pub enum CryptoThresholdMode {
    Distance,
    LonLat,
}

fn transform<T>(x: T, y: T) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
    let xy = x * y;
    let abs_x = x.abs().sqrt();
    let x_pi = x * T::PI();
    let y_pi = y * T::PI();
    let d: T = num!(20.0) * (num!(6.0) * x_pi).sin() + num!(20.0) * (T::TWO * x_pi).sin();

    let mut lat = d;
    let mut lon = d;

    lat += num!(20.0) * (y_pi).sin()
        + num!(40.0) * (y_pi / num!(3.0)).sin()
        + num!(160.0) * (y_pi / num!(12.0)).sin()
        + num!(320.0) * (y_pi / num!(30.0)).sin();
    lon += num!(20.0) * (x_pi).sin()
        + num!(40.0) * (x_pi / num!(3.0)).sin()
        + num!(150.0) * (x_pi / num!(12.0)).sin()
        + num!(300.0) * (x_pi / num!(30.0)).sin();

    lat *= num!(2.0) / num!(3.0);
    lon *= num!(2.0) / num!(3.0);

    lat += num!(-100.0)
        + T::TWO * x
        + num!(3.0) * y
        + num!(0.2) * y.powi(2)
        + num!(0.1) * xy
        + num!(0.2) * abs_x;
    lon +=
        num!(300.0) + x + T::TWO * y + num!(0.1) * x.powi(2) + num!(0.1) * xy + num!(0.1) * abs_x;

    (lon, lat)
}

fn delta<T>(lon: T, lat: T) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
    let (mut d_lon, mut d_lat) = transform(lon - num!(105.0), lat - num!(35.0));
    let rad_lat = lat / num!(180.0) * T::PI();
    let mut magic = (rad_lat).sin();
    let ee = T::krasovsky1940().eccentricity2();
    let earth_r = T::krasovsky1940().semi_major_axis();

    magic = T::ONE - ee * magic * magic;
    let sqrt_magic = (magic).sqrt();
    d_lat = (d_lat * num!(180.0)) / ((earth_r * (T::ONE - ee)) / (magic * sqrt_magic) * T::PI());
    d_lon = (d_lon * num!(180.0)) / (earth_r / sqrt_magic * (rad_lat).cos() * T::PI());
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
/// use pyxis::crypto::*;
/// let p = (BD09_LON, BD09_LAT);
/// let p = bd09_to_gcj02(p.0, p.1);
/// assert_approx_eq!(f64, p.0, GCJ02_LON, epsilon = 1e-6);
/// assert_approx_eq!(f64, p.1, GCJ02_LAT, epsilon = 1e-6);
/// ```
pub fn bd09_to_gcj02<T>(bd09_lon: T, bd09_lat: T) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
    let x_pi = T::PI() * num!(3000.0) / num!(180.0);
    let x = bd09_lon - num!(0.0065);
    let y = bd09_lat - num!(0.006);
    let z = (x.powi(2) + y.powi(2)).sqrt() - num!(0.00002) * (y * x_pi).sin();
    let theta = y.atan2(x) - num!(0.000003) * (x * x_pi).cos();
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
/// use pyxis::crypto::*;
/// let p = (GCJ02_LON, GCJ02_LAT);
/// let p = gcj02_to_wgs84(p.0, p.1);
/// assert_approx_eq!(f64, p.0, WGS84_LON , epsilon = 1e-5);
/// assert_approx_eq!(f64, p.1, WGS84_LAT, epsilon = 1e-7);
/// ```
pub fn gcj02_to_wgs84<T>(gcj02_lon: T, gcj02_lat: T) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
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
/// use pyxis::crypto::*;
/// let p = (BD09_LON, BD09_LAT);
/// let p = bd09_to_wgs84(p.0, p.1);
/// assert_approx_eq!(f64, p.0, WGS84_LON, epsilon = 1e-5);
/// assert_approx_eq!(f64, p.1, WGS84_LAT, epsilon = 1e-5);
/// ```
pub fn bd09_to_wgs84<T>(bd09_lon: T, bd09_lat: T) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
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
/// use pyxis::crypto::*;
/// let p = (GCJ02_LON, GCJ02_LAT);
/// let p = gcj02_to_bd09(p.0, p.1);
/// assert_approx_eq!(f64, p.0, BD09_LON, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, BD09_LAT, epsilon = 1e-17);
/// ```
pub fn gcj02_to_bd09<T>(gcj02_lon: T, gcj02_lat: T) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
    let x_pi = T::PI() * num!(3000.0) / num!(180.0);
    let z =
        (gcj02_lon.powi(2) + gcj02_lat.powi(2)).sqrt() + num!(0.00002) * (gcj02_lat * x_pi).sin();
    let theta = gcj02_lat.atan2(gcj02_lon) + num!(0.000003) * (gcj02_lon * x_pi).cos();
    let bd09_lon = z * (theta).cos() + num!(0.0065);
    let bd09_lat = z * (theta).sin() + num!(0.006);
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
/// use pyxis::crypto::*;
/// let p = (WGS84_LON,WGS84_LAT );
/// let p = wgs84_to_gcj02(p.0, p.1);
/// println!("{:.60},{:.60}",p.0,p.1);
/// assert_approx_eq!(f64, p.0, GCJ02_LON, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, GCJ02_LAT, epsilon = 1e-17);
/// ```
pub fn wgs84_to_gcj02<T>(wgs84_lon: T, wgs84_lat: T) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
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
/// use pyxis::crypto::*;
/// let p = (WGS84_LON,WGS84_LAT );
/// let p = wgs84_to_bd09(p.0, p.1);
/// println!("{:.60},{:.60}",p.0,p.1);
/// assert_approx_eq!(f64, p.0, BD09_LON, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, BD09_LAT,  epsilon = 1e-17);
/// ```
pub fn wgs84_to_bd09<T>(wgs84_lon: T, wgs84_lat: T) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
    let (gcj_lon, gcj_lat) = wgs84_to_gcj02(wgs84_lon, wgs84_lat);
    gcj02_to_bd09(gcj_lon, gcj_lat)
}

/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use tracing_subscriber::layer::SubscriberExt;
/// use tracing_subscriber::util::SubscriberInitExt;
/// use tracing_subscriber::filter::LevelFilter;
/// use pyxis::crypto::*;
/// tracing_subscriber::registry()
///     .with(log_template::terminal_layer(LevelFilter::TRACE))
///     .init();
/// let p = (BD09_LON, BD09_LAT);
/// let p = crypto_exact(
///             p.0,
///             p.1,
///             &bd09_to_wgs84,
///             &wgs84_to_bd09,
///             1e-17,
///             CryptoThresholdMode::LonLat,
///             100,
///         );
/// assert_approx_eq!(f64, p.0, WGS84_LON, epsilon = 1e-13);
/// assert_approx_eq!(f64, p.1, WGS84_LAT, epsilon = 1e-13);
/// ```
///
/// ```
/// use float_cmp::assert_approx_eq;
/// use tracing_subscriber::layer::SubscriberExt;
/// use tracing_subscriber::util::SubscriberInitExt;
/// use tracing_subscriber::filter::LevelFilter;
/// use pyxis::crypto::*;
/// tracing_subscriber::registry()
///     .with(log_template::terminal_layer(LevelFilter::TRACE))
///     .init();
/// let p = (BD09_LON, BD09_LAT);
/// let p = crypto_exact(
///             p.0,
///             p.1,
///             &bd09_to_wgs84,
///             &wgs84_to_bd09,
///             1e-4,
///             CryptoThresholdMode::Distance,
///             100,
///         );
/// assert_approx_eq!(f64, p.0, WGS84_LON, epsilon = 1e-8);
/// assert_approx_eq!(f64, p.1, WGS84_LAT, epsilon = 1e-8);
/// ```
pub fn crypto_exact<T>(
    src_lon: T,
    src_lat: T,
    crypto_fn: &impl Fn(T, T) -> (T, T),
    inv_crypto_fn: &impl Fn(T, T) -> (T, T),
    threshold: T,
    threshold_mode: CryptoThresholdMode,
    max_iter: usize,
) -> (T, T)
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
    let (mut dst_lon, mut dst_lat) = crypto_fn(src_lon, src_lat);
    for _i in 0..max_iter {
        let (tmp_src_lon, tmp_src_lat) = inv_crypto_fn(dst_lon, dst_lat);
        let (d_lon, d_lat) = (src_lon - tmp_src_lon, src_lat - tmp_src_lat);

        let tmp_lon = dst_lon + d_lon;
        let tmp_lat = dst_lat + d_lat;

        tracing::trace!("iteration: {_i}");
        tracing::trace!("dst_lon: {dst_lon}, dst_lat: {dst_lat}");
        tracing::trace!("d_lon: {:.2e}, d_lat: {:.2e}", d_lon, d_lat);
        tracing::trace!(
            "distance: {}",
            haversine_distance(src_lon, src_lat, tmp_lon, tmp_lat)
        );
        if _i == max_iter - 1 {
            tracing::warn!("Exeed max iteration num!ber: {max_iter}");
        }

        match threshold_mode {
            CryptoThresholdMode::Distance
                if haversine_distance(tmp_lon, tmp_lat, dst_lon, dst_lat) < threshold =>
            {
                break;
            }
            CryptoThresholdMode::LonLat
                if (d_lon).abs() < threshold && (d_lat).abs() < threshold =>
            {
                break;
            }
            _ => (),
        }
        (dst_lon, dst_lat) = (tmp_lon, tmp_lat);
    }
    (dst_lon, dst_lat)
}
/// distance calculate the distance between point(lat_a, lon_a) and point(lat_b, lon_b), unit in meter.
pub fn haversine_distance<T>(lon_a: T, lat_a: T, lon_b: T, lat_b: T) -> T
where
    T: GeoFloat + ConstEllipsoid<T> + 'static,
{
    let lat1_rad = lat_a.to_radians();
    let lon1_rad = lon_a.to_radians();
    let lat2_rad = lat_b.to_radians();
    let lon2_rad = lon_b.to_radians();

    let delta_lat = lat2_rad - lat1_rad;
    let delta_lon = lon2_rad - lon1_rad;

    let a = (delta_lat / T::TWO).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / T::TWO).sin().powi(2);
    let c = T::TWO * a.sqrt().atan2((T::ONE - a).sqrt());

    T::wgs84().semi_major_axis() * c
}
#[cfg(test)]
mod test {

    use core::f64;

    use float_cmp::assert_approx_eq;
    use rand::prelude::*;
    use tracing_subscriber::filter::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    use super::*;

    #[test]
    fn test_exact() {
        tracing_subscriber::registry()
            .with(clerk::terminal_layer(LevelFilter::ERROR))
            .init();
        let is_ci = std::env::var("CI").is_ok();
        let mut rng = rand::rng();
        let threshold = 1e-13;
        let mut max_dist: f64 = 0.0;
        let mut max_lonlat: f64 = 0.0;
        let mut all_dist = 0.0;
        let mut all_lonlat = 0.0;
        let count = if is_ci { 10 } else { 10000 };
        for _ in 0..count {
            let wgs: (f64, f64) = (
                rng.random_range(72.004..137.8347),
                rng.random_range(0.8293..55.8271),
            );
            let gcj = wgs84_to_gcj02(wgs.0, wgs.1);
            let bd = wgs84_to_bd09(wgs.0, wgs.1);
            {
                let test_gcj = crypto_exact(
                    bd.0,
                    bd.1,
                    &bd09_to_gcj02,
                    &gcj02_to_bd09,
                    1e-20,
                    CryptoThresholdMode::LonLat,
                    100,
                );
                if (test_gcj.0 - gcj.0).abs() > threshold || (test_gcj.1 - gcj.1).abs() > threshold
                {
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
                max_dist =
                    max_dist.max(haversine_distance(test_gcj.0, test_gcj.1, gcj.0, gcj.1).abs());
                max_lonlat = max_lonlat
                    .max((test_gcj.0 - gcj.0).abs())
                    .max((test_gcj.1 - gcj.1).abs());
                all_dist += haversine_distance(test_gcj.0, test_gcj.1, gcj.0, gcj.1).abs();
                all_lonlat += (test_gcj.0 - gcj.0).abs() + (test_gcj.1 - gcj.1).abs();
                if is_ci {
                    assert_approx_eq!(f64, test_gcj.0, gcj.0, epsilon = threshold);
                    assert_approx_eq!(f64, test_gcj.1, gcj.1, epsilon = threshold);
                }
            }
            {
                let test_wgs = crypto_exact(
                    bd.0,
                    bd.1,
                    &bd09_to_wgs84,
                    &wgs84_to_bd09,
                    1e-20,
                    CryptoThresholdMode::LonLat,
                    100,
                );
                if (test_wgs.0 - wgs.0).abs() > threshold || (test_wgs.1 - wgs.1).abs() > threshold
                {
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
                max_dist =
                    max_dist.max(haversine_distance(test_wgs.0, test_wgs.1, wgs.0, wgs.1).abs());
                max_lonlat = max_lonlat
                    .max((test_wgs.0 - wgs.0).abs())
                    .max((test_wgs.1 - wgs.1).abs());
                all_dist += haversine_distance(test_wgs.0, test_wgs.1, wgs.0, wgs.1).abs();
                all_lonlat += (test_wgs.0 - wgs.0).abs() + (test_wgs.1 - wgs.1).abs();
                if is_ci {
                    assert_approx_eq!(f64, test_wgs.0, wgs.0, epsilon = threshold);
                    assert_approx_eq!(f64, test_wgs.1, wgs.1, epsilon = threshold);
                }
            }
            {
                let test_wgs = crypto_exact(
                    gcj.0,
                    gcj.1,
                    &gcj02_to_wgs84,
                    &wgs84_to_gcj02,
                    1e-20,
                    CryptoThresholdMode::LonLat,
                    100,
                );
                if (test_wgs.0 - wgs.0).abs() > threshold || (test_wgs.1 - wgs.1).abs() > threshold
                {
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
                max_dist =
                    max_dist.max(haversine_distance(test_wgs.0, test_wgs.1, wgs.0, wgs.1).abs());
                max_lonlat = max_lonlat
                    .max((test_wgs.0 - wgs.0).abs())
                    .max((test_wgs.1 - wgs.1).abs());
                all_dist += haversine_distance(test_wgs.0, test_wgs.1, wgs.0, wgs.1).abs();
                all_lonlat += (test_wgs.0 - wgs.0).abs() + (test_wgs.1 - wgs.1).abs();
                if is_ci {
                    assert_approx_eq!(f64, test_wgs.0, wgs.0, epsilon = threshold);
                    assert_approx_eq!(f64, test_wgs.1, wgs.1, epsilon = threshold);
                }
            }
        }
        println!("average distance: {:.2e}", all_dist / count as f64 / 3.0);
        println!("max distance: {:.2e}", max_dist);
        println!("average lonlat: {:.2e}", all_lonlat / count as f64 / 6.0);
        println!("max lonlat: {:.2e}", max_lonlat);
    }
}
