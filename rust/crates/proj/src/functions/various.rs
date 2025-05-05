use std::char;
use std::ffi::CString;

use miette::IntoDiagnostic;

#[cfg(any(feature = "unrecommended", test))]
use crate::check_result;

/// # Various
impl crate::Pj {
    ///Measure internal consistency of a given transformation. The function
    /// performs n round trip transformations starting in either the forward or
    /// reverse direction. Returns the euclidean distance of the starting point
    /// coo and the resulting coordinate after n iterations back and forth.
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_roundtrip>
    #[cfg(any(feature = "unrecommended", test))]
    pub fn roundtrip(
        &self,
        dir: &crate::PjDirection,
        n: i32,
        coord: &mut crate::PjCoord,
    ) -> miette::Result<f64> {
        let distance = unsafe { proj_sys::proj_roundtrip(self.pj, i32::from(dir), n, coord) };
        check_result!(self);
        Ok(distance)
    }

    ///Calculate various cartographic properties, such as scale factors,
    /// angular distortion and meridian convergence. Depending on the underlying
    /// projection values will be calculated either numerically (default) or
    /// analytically.
    ///
    ///Starting with PROJ 8.2, the P object can be a projected CRS, for example
    /// instantiated from a EPSG CRS code. The factors computed will be those of
    /// the map projection implied by the transformation from the base
    /// geographic CRS of the projected CRS to the projected CRS. Starting with
    /// PROJ 9.6, to improve performance on repeated calls on a projected CRS
    /// object, the above steps will modify the internal state of the provided P
    /// object, and thus calling this function concurrently from multiple
    /// threads on the same P object will no longer be supported.
    ///
    ///The input geodetic coordinate lp should be such that lp.lam is the
    /// longitude in radian, and lp.phi the latitude in radian (thus
    /// independently of the definition of the base CRS, if P is a projected
    /// CRS).
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_factors>
    #[cfg(any(feature = "unrecommended", test))]
    pub fn factors(&self, coord: crate::PjCoord) -> miette::Result<crate::PjFactors> {
        let factor = unsafe { proj_sys::proj_factors(self.pj, coord) };
        match self.errno() {
            crate::PjErrorCode::ProjSuccess => (),
            crate::PjErrorCode::CoordTransfmOutsideProjectionDomain => (),
            _ => {
                check_result!(self);
            }
        };

        let factor = crate::PjFactors::new(
            factor.meridional_scale,
            factor.parallel_scale,
            factor.areal_scale,
            factor.angular_distortion,
            factor.meridian_parallel_angle,
            factor.meridian_convergence,
            factor.tissot_semimajor,
            factor.tissot_semiminor,
            factor.dx_dlam,
            factor.dx_dphi,
            factor.dy_dlam,
            factor.dy_dphi,
        );
        Ok(factor)
    }

    ///Check if an operation expects input in radians or not.
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_angular_input>
    pub fn angular_input(&self, dir: &crate::PjDirection) -> miette::Result<bool> {
        let result = unsafe { proj_sys::proj_angular_input(self.pj, i32::from(dir)) } != 0;
        Ok(result)
    }

    ///Check if an operation returns output in radians or not.
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_angular_output>
    pub fn angular_output(&self, dir: &crate::PjDirection) -> miette::Result<bool> {
        let result = unsafe { proj_sys::proj_angular_output(self.pj, i32::from(dir)) } != 0;
        Ok(result)
    }

    ///Check if an operation expects input in degrees or not.
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_degree_input>
    pub fn degree_input(&self, dir: &crate::PjDirection) -> miette::Result<bool> {
        let result = unsafe { proj_sys::proj_degree_input(self.pj, i32::from(dir)) } != 0;
        Ok(result)
    }

    ///Check if an operation returns output in degrees or not.
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_degree_output>
    pub fn degree_output(&self, dir: &crate::PjDirection) -> miette::Result<bool> {
        let result = unsafe { proj_sys::proj_degree_output(self.pj, i32::from(dir)) } != 0;
        Ok(result)
    }
}

///Initializer for the PJ_COORD union. The function is shorthand for the
/// otherwise convoluted assignment.
///
///# References
///<https://proj.org/en/stable/development/reference/functions.html#c.proj_coord>
#[cfg(any(feature = "unrecommended", test))]
pub fn coord(x: f64, y: f64, z: f64, t: f64) -> proj_sys::PJ_COORD {
    unsafe { proj_sys::proj_coord(x, y, z, t) }
}

#[deprecated(note = "Use `f64::to_radians(self)` instead")]
fn _torad() { unimplemented!() }

#[deprecated(note = "Use `f64::to_degrees(self)` instead")]
fn _todeg() { unimplemented!() }

pub fn dmstor() -> f64 { unimplemented!() }

#[deprecated(note = "Use `rtodms2()` instead.")]
pub fn rtodms(_r: f64, _pos: char, _neg: char) -> String { unimplemented!() }
///Convert radians to string representation of degrees, minutes and seconds.
///
/// # References
///<https://proj.org/en/stable/development/reference/functions.html#c.proj_rtodms2>
pub fn rtodms2(r: f64, pos: char, neg: char) -> miette::Result<String> {
    let dms = CString::new("xxxdxxmxx.xxs ").into_diagnostic()?;
    let ptr =
        unsafe { proj_sys::proj_rtodms2(dms.as_ptr().cast_mut(), 14, r, pos as i32, neg as i32) };
    Ok(crate::c_char_to_string(ptr))
}

#[cfg(test)]
mod test {
    use std::f64::consts::{FRAC_PI_2, PI};

    use float_cmp::assert_approx_eq;

    use crate::IPjCoord;

    #[test]
    fn test_roundtrip() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create_crs_to_crs("+proj=tmerc +lat_0=0 +lon_0=75 +k=1 +x_0=13500000 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs","EPSG:4326",  &crate::PjArea::default())?;
        let mut coord =
            crate::array4_to_pj_coord((5877537.151800396, 4477291.358855194).to_array4())?;
        let distance = pj.roundtrip(&crate::PjDirection::Fwd, 10000, &mut coord)?;
        println!("{:?}", unsafe { coord.xy.x });
        println!("{:?}", unsafe { coord.xy.y });
        assert_approx_eq!(f64, distance, 0.023350762947799957, epsilon = 1e-6);
        Ok(())
    }
    #[test]
    fn test_factors() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:3857", &crate::PjArea::default())?;
        let factor = pj.factors(crate::array4_to_pj_coord(
            (12.0f64.to_radians(), 55.0f64.to_radians()).to_array4(),
        )?)?;

        println!("{:?}", factor);

        assert_approx_eq!(
            f64,
            factor.meridional_scale().clone(),
            111315.45155747599,
            epsilon = 1e-6
        );
        assert_approx_eq!(
            f64,
            factor.parallel_scale().clone(),
            193644.51017869517,
            epsilon = 1e-6
        );
        assert_approx_eq!(
            f64,
            factor.areal_scale().clone(),
            -21555626092.167713,
            epsilon = 1.0
        );

        assert_approx_eq!(f64, factor.angular_distortion().clone(), PI, epsilon = 1e-6);
        assert_approx_eq!(
            f64,
            factor.meridian_parallel_angle().clone(),
            -FRAC_PI_2,
            epsilon = 1e-6
        );
        assert_approx_eq!(
            f64,
            factor.meridian_convergence().clone(),
            -FRAC_PI_2,
            epsilon = 1e-6
        );

        assert_approx_eq!(
            f64,
            factor.tissot_semimajor().clone(),
            193644.51017869514,
            epsilon = 1e-6
        );
        assert_approx_eq!(
            f64,
            factor.tissot_semiminor().clone(),
            -111315.45155747602,
            epsilon = 1e-6
        );

        assert_approx_eq!(
            f64,
            factor.dx_dlam().clone(),
            3.6379788070917124e-7,
            epsilon = 1e-6
        );
        assert_approx_eq!(
            f64,
            factor.dx_dphi().clone(),
            111319.49079353943,
            epsilon = 1e-6
        );
        assert_approx_eq!(
            f64,
            factor.dy_dlam().clone(),
            111320.23452373686,
            epsilon = 1e-6
        );
        assert_approx_eq!(f64, factor.dy_dphi().clone(), 0.0, epsilon = 1e-6);

        Ok(())
    }

    #[test]
    fn test_factors_fail() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create("EPSG:4326")?;
        let factor = pj.factors(crate::array4_to_pj_coord(
            (12.0f64.to_radians(), 55.0f64.to_radians()).to_array4(),
        )?);
        assert!(factor.is_err());
        Ok(())
    }
    #[test]
    fn test_coor() -> miette::Result<()> {
        super::coord(1.0, 2.0, 3.0, 4.0);
        Ok(())
    }

    #[test]
    fn test_rtodms2() -> miette::Result<()> {
        let dms = super::rtodms2(30.123456789f64.to_radians(), 'E', 'W')?;
        assert_eq!(dms, "30d7'24.444\"E");
        Ok(())
    }
    #[test]
    fn test_input_output_angle_format() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        assert!(!(pj.angular_input(&crate::PjDirection::Fwd)?));
        assert!(!(pj.angular_output(&crate::PjDirection::Fwd)?));
        assert!(pj.degree_input(&crate::PjDirection::Fwd)?);
        assert!(!(pj.degree_output(&crate::PjDirection::Fwd)?));
        Ok(())
    }
}
