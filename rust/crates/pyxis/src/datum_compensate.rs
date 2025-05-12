use crate::GeoFloat;
/// Converts projected XY coordinates from the height compensation plane to
/// the sea level plane.
///
/// # Arguments
///
/// - `xc`, `yc`: Coordinates on the height compensation plane (in meters).
/// - `hb`: Elevation of the height compensation plane (in meters).
/// - `radius`: Radius of the Earth (in meters).
/// - `x0`, `y0`: Coordinate system origin (in meters).
///
/// # Returns
///
/// A tuple containing the projected XY coordinates of the sea level plane
/// (in meters).
///
///
/// # References
/// - 杨元兴. (2008). 抵偿高程面的选择与计算. 城市勘测 (02), 72-74.
///
/// # Examples
/// ```
/// use float_cmp::assert_approx_eq;
/// let p = (469704.6693, 2821940.796);
/// let processor = pyxis::DatumCompensate::new(400.0, 6_378_137.0, 500_000.0, 0.0);
/// let p = processor.transfrom(p.0, p.1);
/// assert_approx_eq!(f64, p.0, 469706.56912942487, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, 2821763.831232311, epsilon = 1e-17);
/// ```
pub struct DatumCompensate<T: GeoFloat> {
    x0: T,
    y0: T,
    factor: T,
}
impl<T: GeoFloat> DatumCompensate<T> {
    pub fn new(hb: T, radius: T, x0: T, y0: T) -> Self {
        let ratio = hb / radius;
        let factor = ratio / (T::ONE + ratio);
        Self { x0, y0, factor }
    }

    pub fn transfrom(&self, xc: T, yc: T) -> (T, T)
    where
        T: GeoFloat,
    {
        let xc = xc - self.factor * (xc - self.x0);
        let yc = yc - self.factor * (yc - self.y0);
        (xc, yc)
    }
}
