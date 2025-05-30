use crate::{ToCoord, check_result};
///# Distances
impl crate::Proj<'_> {
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_lp_dist>
    pub fn lp_dist(&self, a: impl crate::ICoord, b: impl crate::ICoord) -> miette::Result<f64> {
        let dist = unsafe { proj_sys::proj_lp_dist(self.ptr(), a.to_coord()?, b.to_coord()?) };
        check_result!(self);
        if dist.is_nan() {
            miette::bail!(
                help = "Check Pj object and make sure input coordinates are in radians.",
                "Distance calculation failed."
            )
        }
        Ok(dist)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_lpz_dist>
    pub fn lpz_dist(&self, a: impl crate::ICoord, b: impl crate::ICoord) -> miette::Result<f64> {
        let dist = unsafe { proj_sys::proj_lpz_dist(self.ptr(), a.to_coord()?, b.to_coord()?) };
        check_result!(self);
        if dist.is_nan() {
            miette::bail!(
                help = "Check Pj object and make sure input coordinates are in radians.",
                "Distance calculation failed."
            )
        }
        Ok(dist)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_xy_dist>
    pub fn geod(&self, a: impl crate::ICoord, b: impl crate::ICoord) -> miette::Result<(f64, f64)> {
        let dist = unsafe { proj_sys::proj_geod(self.ptr(), a.to_coord()?, b.to_coord()?) };
        check_result!(self);
        let (dist, reversed_azimuth) = unsafe { (dist.lp.lam, dist.lp.phi) };
        if dist.is_nan() || reversed_azimuth.is_nan() {
            miette::bail!(
                help = "Check Pj object and make sure input coordinates are in radians.",
                "Distance calculation failed."
            )
        }
        Ok((dist, reversed_azimuth))
    }
}
/// Calculate 2-dimensional euclidean between two projected coordinates.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_xy_dist>
pub fn xy_dist(a: impl crate::ICoord, b: impl crate::ICoord) -> miette::Result<f64> {
    Ok(unsafe { proj_sys::proj_xy_dist(a.to_coord()?, b.to_coord()?) })
}
/// Calculate 3-dimensional euclidean between two projected coordinates.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_xyz_dist>
pub fn xyz_dist(a: impl crate::ICoord, b: impl crate::ICoord) -> miette::Result<f64> {
    Ok(unsafe { proj_sys::proj_xyz_dist(a.to_coord()?, b.to_coord()?) })
}
#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_lp_dist() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let dist = pj.lp_dist(
            (1.0f64.to_radians(), 2.0f64.to_radians()),
            (3.0f64.to_radians(), 4.0f64.to_radians()),
        )?;
        assert_approx_eq!(f64, dist, 313588.39721259556);
        Ok(())
    }
    #[test]
    fn test_lpz_dist() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let dist = pj.lpz_dist(
            (118.0f64.to_radians(), 30.0f64.to_radians(), 1.0),
            (119.0f64.to_radians(), 40.0f64.to_radians(), 2000.0),
        )?;
        assert_approx_eq!(f64, dist, 1113143.341157136);
        Ok(())
    }
    #[test]
    fn test_geod() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let (dist, reversed_azimuth) = pj.geod(
            (1.0f64.to_radians(), 2.0f64.to_radians()),
            (3.0f64.to_radians(), 4.0f64.to_radians()),
        )?;
        assert_approx_eq!(f64, dist, 313588.39721259556);
        assert_approx_eq!(f64, reversed_azimuth, 45.10460545587798);
        Ok(())
    }
    #[test]
    fn test_xy_dist() -> miette::Result<()> {
        let dist = super::xy_dist((1.0, 2.0), (4.0, 6.0))?;
        assert_approx_eq!(f64, dist, 5.0);
        Ok(())
    }
    #[test]
    fn test_xyz_dist() -> miette::Result<()> {
        let dist = super::xyz_dist((1.0, 2.0, 1.0), (2.0, 4.0, 5.0))?;
        assert_approx_eq!(f64, dist, 21.0f64.sqrt());
        Ok(())
    }
}
