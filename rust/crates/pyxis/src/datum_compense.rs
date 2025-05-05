use crate::GeoFloat;
pub trait IDatumCompenseParms<T: GeoFloat> {
    fn x0(&self) -> T;
    fn y0(&self) -> T;
    fn factor(&self) -> T;
}
pub struct DatumCompenseParms<T: GeoFloat> {
    x0: T,
    y0: T,
    factor: T,
}
impl<T: GeoFloat> DatumCompenseParms<T> {
    pub fn new(hb: T, radius: T, x0: T, y0: T) -> Self {
        let ratio = hb / radius;
        let factor = ratio / (T::ONE + ratio);
        Self { x0, y0, factor }
    }
}
impl<T: GeoFloat> IDatumCompenseParms<T> for DatumCompenseParms<T> {
    fn x0(&self) -> T { self.x0 }
    fn y0(&self) -> T { self.y0 }
    fn factor(&self) -> T { self.factor }
}
/// Converts projected XY coordinates from the height compensation plane to the
/// sea level plane.
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
/// A tuple containing the projected XY coordinates of the sea level plane (in
/// meters).
///
///
/// # References
/// - 杨元兴. (2008). 抵偿高程面的选择与计算. 城市勘测 (02), 72-74.
///
/// # Examples
/// ```
/// use float_cmp::assert_approx_eq;
/// let p =(469704.6693, 2821940.796);
/// let parms=pyxis::DatumCompenseParms::new(400.0, 6_378_137.0, 500_000.0, 0.0);
/// let p = pyxis::datum_compense(p.0, p.1, &parms);
/// assert_approx_eq!(f64, p.0, 469706.56912942487, epsilon = 1e-17);
/// assert_approx_eq!(f64, p.1, 2821763.831232311, epsilon = 1e-17);
/// ```
pub fn datum_compense<T>(xc: T, yc: T, parms: &impl IDatumCompenseParms<T>) -> (T, T)
where
    T: GeoFloat,
{
    let xc = xc - parms.factor() * (xc - parms.x0());
    let yc = yc - parms.factor() * (yc - parms.y0());
    (xc, yc)
}
