use core::fmt;
use core::str::FromStr;
extern crate alloc;
use alloc::string::{String, ToString};

/// # References
/// - https://github.com/googollee/eviltransform/blob/master/rust/src/lib.rs
/// - https://github.com/billtian/wgtochina_lb-php/tree/master
/// - https://github.com/Leask/EvilTransform
/// - https://github.com/wandergis/coordtransform
/// - https://blog.csdn.net/coolypf/article/details/8569813
/// - https://github.com/Artoria2e5/PRCoords/blob/master/js/PRCoords.js
use crate::primitive::{GeoFloat, num};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CryptoSpace {
    BD09,
    GCJ02,
    WGS84,
}

impl FromStr for CryptoSpace {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "BD09" => Ok(Self::BD09),
            "GCJ02" => Ok(Self::GCJ02),
            "WGS84" => Ok(Self::WGS84),
            _ => Err("".to_string()),
        }
    }
}
impl fmt::Display for CryptoSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BD09 => write!(f, "BD09"),
            Self::GCJ02 => write!(f, "GCJ02"),
            Self::WGS84 => write!(f, "WGS84"),
        }
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum CryptoThresholdMode<T>
where
    T: GeoFloat + 'static,
{
    Distance { semi_major_axis: T },
    LonLat,
}

impl<T> fmt::Display for CryptoThresholdMode<T>
where
    T: GeoFloat + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Distance { semi_major_axis } => {
                write!(f, "Distance{{semi_major_axis: {}}}", semi_major_axis)
            }
            Self::LonLat => write!(f, "LonLat"),
        }
    }
}
fn transform<T>(x: T, y: T) -> (T, T)
where
    T: GeoFloat + 'static,
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
    T: GeoFloat + 'static,
{
    let (mut d_lon, mut d_lat) = transform(lon - num!(105.0), lat - num!(35.0));
    let rad_lat = lat / num!(180.0) * T::PI();
    let mut magic = (rad_lat).sin();
    let ee = num!(0.006_694_380_022_900_787);
    let earth_r = num!(6378137.0);

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
/// A tuple `(lon, lat)` representing the coordinates in the `GCJ02` coordinate
/// system:
/// - `lon`: Longitude in the `GCJ02` coordinate system.
/// - `lat`: Latitude in the `GCJ02` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::crypto::*;
/// clerk::init_log_with_level(clerk::LevelFilter::TRACE);
/// let p = (120.0, 30.0);
/// let p = bd09_to_gcj02(p.0, p.1);
/// println!("{},{}", p.0, p.1);
/// ```
pub fn bd09_to_gcj02<T>(bd09_lon: T, bd09_lat: T) -> (T, T)
where
    T: GeoFloat + 'static,
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
/// A tuple `(lon, lat)` representing the coordinates in the `WGS84` coordinate
/// system:
/// - `lon`: Longitude in the `WGS84` coordinate system.
/// - `lat`: Latitude in the `WGS84` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::crypto::*;
/// clerk::init_log_with_level(clerk::LevelFilter::TRACE);
/// let p = (120.0, 30.0);
/// let p = gcj02_to_wgs84(p.0, p.1);
/// println!("{},{}", p.0, p.1);
/// ```
pub fn gcj02_to_wgs84<T>(gcj02_lon: T, gcj02_lat: T) -> (T, T)
where
    T: GeoFloat + 'static,
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
/// A tuple `(lon, lat)` representing the coordinates in the `WGS84` coordinate
/// system:
/// - `lon`: Longitude in the `WGS84` coordinate system.
/// - `lat`: Latitude in the `WGS84` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::crypto::*;
/// clerk::init_log_with_level(clerk::LevelFilter::TRACE);
/// let p = (120.0, 30.0);
/// let p = bd09_to_wgs84(p.0, p.1);
/// println!("{},{}", p.0, p.1);
/// ```
pub fn bd09_to_wgs84<T>(bd09_lon: T, bd09_lat: T) -> (T, T)
where
    T: GeoFloat + 'static,
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
/// A tuple `(lon, lat)` representing the coordinates in the `BD09` coordinate
/// system:
/// - `lon`: Longitude in the `BD09` coordinate system.
/// - `lat`: Latitude in the `BD09` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::crypto::*;
/// clerk::init_log_with_level(clerk::LevelFilter::TRACE);
/// let p = (120.0, 30.0);
/// let p = gcj02_to_bd09(p.0, p.1);
/// println!("{},{}", p.0, p.1);
/// ```
pub fn gcj02_to_bd09<T>(gcj02_lon: T, gcj02_lat: T) -> (T, T)
where
    T: GeoFloat + 'static,
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
/// A tuple `(lon, lat)` representing the coordinates in the `GCJ02` coordinate
/// system:
/// - `lon`: Longitude in the `GCJ02` coordinate system.
/// - `lat`: Latitude in the `GCJ02` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::crypto::*;
/// clerk::init_log_with_level(clerk::LevelFilter::TRACE);
/// let p = (120.0, 30.0);
/// let p = wgs84_to_gcj02(p.0, p.1);
/// println!("{},{}", p.0, p.1);
/// ```
pub fn wgs84_to_gcj02<T>(wgs84_lon: T, wgs84_lat: T) -> (T, T)
where
    T: GeoFloat + 'static,
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
/// A tuple `(lon, lat)` representing the coordinates in the `WGS84` coordinate
/// system:
/// - `wgs84_lon`: Longitude in the `WGS84` coordinate system.
/// - `wgs84_lat`: Latitude in the `WGS84` coordinate system.
///
/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::crypto::*;
/// clerk::init_log_with_level(clerk::LevelFilter::TRACE);
/// let p = (120.0, 30.0);
/// let p = wgs84_to_bd09(p.0, p.1);
/// println!("{},{}", p.0, p.1);
/// ```
pub fn wgs84_to_bd09<T>(wgs84_lon: T, wgs84_lat: T) -> (T, T)
where
    T: GeoFloat + 'static,
{
    let (gcj_lon, gcj_lat) = wgs84_to_gcj02(wgs84_lon, wgs84_lat);
    gcj02_to_bd09(gcj_lon, gcj_lat)
}

/// # Example
/// ```
/// use float_cmp::assert_approx_eq;
/// use pyxis::crypto::*;
/// clerk::init_log_with_level(clerk::LevelFilter::TRACE);
/// let p = (120.0, 30.0);
/// let p = crypto_exact(
///     p.0,
///     p.1,
///     &bd09_to_wgs84,
///     &wgs84_to_bd09,
///     1e-17,
///     &CryptoThresholdMode::LonLat,
///     100,
/// );
/// println!("{},{}", p.0, p.1);
/// ```
pub fn crypto_exact<T>(
    src_lon: T,
    src_lat: T,
    crypto_fn: &impl Fn(T, T) -> (T, T),
    inv_crypto_fn: &impl Fn(T, T) -> (T, T),
    threshold: T,
    threshold_mode: &CryptoThresholdMode<T>,
    max_iter: usize,
) -> (T, T)
where
    T: GeoFloat + 'static,
{
    let (mut dst_lon, mut dst_lat) = inv_crypto_fn(src_lon, src_lat);
    for _i in 0..max_iter {
        let (tmp_src_lon, tmp_src_lat) = crypto_fn(dst_lon, dst_lat);
        let (d_lon, d_lat) = (src_lon - tmp_src_lon, src_lat - tmp_src_lat);

        let tmp_lon = dst_lon + d_lon;
        let tmp_lat = dst_lat + d_lat;
        #[cfg(debug_assertions)]
        {
            clerk::trace!("iteration: {_i}");
            clerk::trace!("dst_lon: {dst_lon}, dst_lat: {dst_lat}");
            clerk::trace!("d_lon: {:.2e}, d_lat: {:.2e}", d_lon, d_lat);
        }
        match threshold_mode {
            CryptoThresholdMode::Distance { semi_major_axis }
                if haversine_distance(tmp_lon, tmp_lat, dst_lon, dst_lat, *semi_major_axis)
                    < threshold =>
            {
                return (dst_lon, dst_lat);
            }
            CryptoThresholdMode::LonLat
                if (d_lon).abs() < threshold && (d_lat).abs() < threshold =>
            {
                return (dst_lon, dst_lat);
            }
            _ => (),
        }
        (dst_lon, dst_lat) = (tmp_lon, tmp_lat);
    }
    clerk::warn!("Exceed max iteration num!ber: {max_iter}");
    (dst_lon, dst_lat)
}
/// distance calculate the distance between point(lat_a, lon_a) and point(lat_b,
/// lon_b), unit in meter.
pub fn haversine_distance<T>(lon_a: T, lat_a: T, lon_b: T, lat_b: T, semi_major_axis: T) -> T
where
    T: GeoFloat + 'static,
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

    semi_major_axis * c
}
#[cfg(test)]
mod test {
    extern crate std;

    use float_cmp::assert_approx_eq;
    use proptest::prelude::*;

    use super::*;
    #[rstest::rstest]
    #[case("wgs84", "gcj02", wgs84_to_gcj02)]
    #[case("gcj02", "wgs84", gcj02_to_wgs84)]
    #[case("gcj02", "bd09", gcj02_to_bd09)]
    #[case("wgs84", "bd09", wgs84_to_bd09)]
    #[case("bd09", "gcj02", bd09_to_gcj02)]
    #[case("bd09", "wgs84", bd09_to_wgs84)]
    fn test_crypto(
        #[case] from: &str,
        #[case] to: &str,
        #[case] f: impl Fn(f64, f64) -> (f64, f64),
    ) {
        let lon = 120.0;
        let lat = 30.0;
        let p = f(lon, lat);
        insta::assert_debug_snapshot!(std::format!("test_crypto-{from}-{to}"), p);
    }

    proptest! {
        #[test]
        fn test_wgs84_gcj02(lon in 72.004..137.8347f64,lat in 0.8293..55.8271f64) {
            let gcj = wgs84_to_gcj02(lon, lat);
            let out = crypto_exact(gcj.0, gcj.1, &wgs84_to_gcj02, &gcj02_to_wgs84, 1e-20, &CryptoThresholdMode::LonLat, 100);
            assert_approx_eq!(f64,out.0, lon, epsilon = 1e-13);
            assert_approx_eq!(f64,out.1, lat, epsilon = 1e-13);
        }
        #[test]
        fn test_wgs84_bd09(lon in 72.004..137.8347f64,lat in 0.8293..55.8271f64) {
            let bd = wgs84_to_bd09(lon, lat);
            let out = crypto_exact(bd.0, bd.1, &wgs84_to_bd09, &bd09_to_wgs84, 1e-20, &CryptoThresholdMode::LonLat, 100);
            assert_approx_eq!(f64,out.0, lon, epsilon = 1e-13);
            assert_approx_eq!(f64,out.1, lat, epsilon = 1e-13);
        }
        #[test]
        fn test_gcj02_bd09(lon in 72.004..137.8347f64,lat in 0.8293..55.8271f64) {
            let bd = gcj02_to_bd09(lon, lat);
            let out = crypto_exact(bd.0, bd.1, &gcj02_to_bd09, &bd09_to_gcj02, 1e-20, &CryptoThresholdMode::LonLat, 100);
            assert_approx_eq!(f64,out.0, lon, epsilon = 1e-13);
            assert_approx_eq!(f64,out.1, lat, epsilon = 1e-13);
        }
    }
}
