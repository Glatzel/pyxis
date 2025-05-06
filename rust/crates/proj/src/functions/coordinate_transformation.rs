use crate::check_result;
// region:Coordinate transformation
impl crate::Pj {
    /// <div class="warning">Available on <b>crate feature</b>
    /// <code>unrecommended</code> only.</div>
    ///
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_trans>
    #[cfg(any(feature = "unrecommended", test))]
    pub fn trans(
        &self,
        direction: crate::PjDirection,
        coord: crate::data_types::PjCoord,
    ) -> miette::Result<crate::data_types::PjCoord> {
        let out_coord = unsafe { proj_sys::proj_trans(self.pj, i32::from(direction), coord) };
        check_result!(self);
        Ok(out_coord)
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_get_last_used_operation>
    fn _get_last_used_operation(&self) -> Self { unimplemented!() }

    /// # Safety
    /// If x,y is not null pointer.
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_generic>
    pub unsafe fn trans_generic(
        &self,
        direction: crate::PjDirection,
        x: *mut f64,
        sx: usize,
        nx: usize,
        y: *mut f64,
        sy: usize,
        ny: usize,
        z: *mut f64,
        sz: usize,
        nz: usize,
        t: *mut f64,
        st: usize,
        nt: usize,
    ) -> miette::Result<usize> {
        let result = unsafe {
            proj_sys::proj_trans_generic(
                self.pj,
                i32::from(direction),
                x,
                sx,
                nx,
                y,
                sy,
                ny,
                z,
                sz,
                nz,
                t,
                st,
                nt,
            )
        };
        check_result!(self);
        Ok(result)
    }

    /// <div class="warning">Available on <b>crate feature</b>
    /// <code>unrecommended</code> only.</div>
    ///
    ///  # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_array>
    #[cfg(any(feature = "unrecommended", test))]
    pub fn trans_array(
        &self,
        direction: crate::PjDirection,
        coord: &mut [crate::data_types::PjCoord],
    ) -> miette::Result<&Self> {
        let code = unsafe {
            proj_sys::proj_trans_array(
                self.pj,
                i32::from(direction),
                coord.len(),
                coord.as_mut_ptr(),
            )
        };

        check_result!(self, code);
        Ok(self)
    }
}

impl crate::PjContext {
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds>
    pub fn trans_bounds(
        &self,
        p: &crate::Pj,
        direction: crate::PjDirection,
        xmin: f64,
        ymin: f64,
        xmax: f64,
        ymax: f64,
        out_xmin: &mut f64,
        out_ymin: &mut f64,
        out_xmax: &mut f64,
        out_ymax: &mut f64,
        densify_pts: i32,
    ) -> miette::Result<&Self> {
        let code = unsafe {
            proj_sys::proj_trans_bounds(
                self.ctx,
                p.pj,
                i32::from(direction),
                xmin,
                ymin,
                xmax,
                ymax,
                out_xmin,
                out_ymin,
                out_xmax,
                out_ymax,
                densify_pts,
            )
        };
        if code != 1 {
            miette::bail!("Failures encountered.")
        }
        Ok(self)
    }

    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds_3D>
    pub fn trans_bounds_3d(
        &self,
        p: &crate::Pj,
        direction: crate::PjDirection,
        xmin: f64,
        ymin: f64,
        zmin: f64,
        xmax: f64,
        ymax: f64,
        zmax: f64,
        out_xmin: &mut f64,
        out_ymin: &mut f64,
        out_zmin: &mut f64,
        out_xmax: &mut f64,
        out_ymax: &mut f64,
        out_zmax: &mut f64,
        densify_pts: i32,
    ) -> miette::Result<&Self> {
        let code = unsafe {
            proj_sys::proj_trans_bounds_3D(
                self.ctx,
                p.pj,
                i32::from(direction),
                xmin,
                ymin,
                zmin,
                xmax,
                ymax,
                zmax,
                out_xmin,
                out_ymin,
                out_zmin,
                out_xmax,
                out_ymax,
                out_zmax,
                densify_pts,
            )
        };
        if code != 1 {
            miette::bail!("Failures encountered.")
        }
        Ok(self)
    }
}
#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;

    use crate::data_types::{PjCoord, PjXy};

    #[test]
    fn test_trans() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let coord = PjCoord {
            xy: PjXy { x: 120.0, y: 30.0 },
        };
        let coord = pj.trans(crate::PjDirection::Fwd, coord)?;

        assert_eq!(unsafe { coord.xy.x }, 19955590.73888901);
        assert_eq!(unsafe { coord.xy.y }, 3416780.562127255);
        Ok(())
    }

    #[test]
    fn test_trans_array() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [
            PjCoord {
                xy: PjXy { x: 120.0, y: 30.0 },
            },
            PjCoord {
                xy: PjXy { x: 50.0, y: -80.0 },
            },
        ];
        pj.trans_array(crate::PjDirection::Fwd, &mut coord)?;
        assert_eq!(unsafe { coord[0].xy.x }, 19955590.73888901);
        assert_eq!(unsafe { coord[0].xy.y }, 3416780.562127255);
        assert_eq!(unsafe { coord[1].xy.x }, 17583572.872089125);
        assert_eq!(unsafe { coord[1].xy.y }, -9356989.97994042);
        Ok(())
    }

    #[test]
    fn test_trans_bounds() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let xmin = 0.0;
        let ymin = 1.0;
        let xmax = 20.0;
        let ymax = 30.0;
        let mut out_xmin = f64::default();
        let mut out_ymin = f64::default();
        let mut out_xmax = f64::default();
        let mut out_ymax = f64::default();

        ctx.trans_bounds(
            &pj,
            crate::PjDirection::Fwd,
            xmin,
            ymin,
            xmax,
            ymax,
            &mut out_xmin,
            &mut out_ymin,
            &mut out_xmax,
            &mut out_ymax,
            21,
        )?;
        assert_approx_eq!(f64, out_xmin, 2297280.4262236636);
        assert_approx_eq!(f64, out_ymin, 6639816.584496002);
        assert_approx_eq!(f64, out_xmax, 10788961.870597329);
        assert_approx_eq!(f64, out_ymax, 19555124.881683525);
        Ok(())
    }

    #[test]
    fn test_trans_bounds_3d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let xmin = 0.0;
        let ymin = 1.0;
        let zmin = 1.0;
        let xmax = 20.0;
        let ymax = 30.0;
        let zmax = 3.0;
        let mut out_xmin = f64::default();
        let mut out_ymin = f64::default();
        let mut out_zmin = f64::default();
        let mut out_xmax = f64::default();
        let mut out_ymax = f64::default();
        let mut out_zmax = f64::default();

        ctx.trans_bounds_3d(
            &pj,
            crate::PjDirection::Fwd,
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
            &mut out_xmin,
            &mut out_ymin,
            &mut out_zmin,
            &mut out_xmax,
            &mut out_ymax,
            &mut out_zmax,
            21,
        )?;
        assert_approx_eq!(f64, out_xmin, 2297280.4262236636);
        assert_approx_eq!(f64, out_ymin, 6639816.584496002);
        assert_approx_eq!(f64, out_zmin, 1.0);
        assert_approx_eq!(f64, out_xmax, 10788961.870597329);
        assert_approx_eq!(f64, out_ymax, 19555124.881683525);
        assert_approx_eq!(f64, out_zmax, 3.0);
        Ok(())
    }
}
