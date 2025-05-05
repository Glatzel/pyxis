use std::char;
use std::ffi::CString;

use miette::IntoDiagnostic;

use crate::{array4_to_pj_coord, check_result};

/// # Various
impl crate::Pj {
    pub fn roundtrip(&self) { unimplemented!() }
    pub fn factors<T>(&self, coord: T) -> miette::Result<crate::PjFactors>
    where
        T: crate::IPjCoord,
    {
        let factor =
            unsafe { proj_sys::proj_factors(self.pj, array4_to_pj_coord(coord.to_array4())?) };
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
    #[test]
    fn test_factors() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:3857", &crate::PjArea::default())?;
        let factor = pj.factors((12.0f64.to_radians(), 55.0f64.to_radians()))?;

        println!("{:?}", factor);

        assert_eq!(factor.meridional_scale(), &111315.45155747599);
        assert_eq!(factor.parallel_scale(), &193644.51017869517);
        assert_eq!(factor.areal_scale(), &-21555626092.167713);

        assert_eq!(factor.angular_distortion(), &3.141592653589793);
        assert_eq!(factor.meridian_parallel_angle(), &-1.5707963267948966);
        assert_eq!(factor.meridian_convergence(), &-1.5707963267948966);

        assert_eq!(factor.tissot_semimajor(), &193644.51017869514);
        assert_eq!(factor.tissot_semiminor(), &-111315.45155747602);

        assert_eq!(factor.dx_dlam(), &3.6379788070917124e-7);
        assert_eq!(factor.dx_dphi(), &111319.49079353943);
        assert_eq!(factor.dy_dlam(), &111320.23452373686);
        assert_eq!(factor.dy_dphi(), &0.0);

        Ok(())
    }
    #[test]
    fn test_factors_fail() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create("EPSG:4326")?;
        let factor = pj.factors((12.0f64.to_radians(), 55.0f64.to_radians()));
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
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        assert_eq!(pj.angular_input(&crate::PjDirection::PjFwd)?, false);
        assert_eq!(pj.angular_output(&crate::PjDirection::PjFwd)?, false);
        assert_eq!(pj.degree_input(&crate::PjDirection::PjFwd)?, true);
        assert_eq!(pj.degree_output(&crate::PjDirection::PjFwd)?, false);
        Ok(())
    }
}
