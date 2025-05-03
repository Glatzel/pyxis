use crate::check_pj_result;

// region:Coordinate transformation
/// # References
///<https://proj.org/en/stable/development/reference/functions.html#coordinate-transformation>
impl crate::Pj {
    ///Not suggested
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans>
    pub fn trans<T>(&self, direction: crate::PjDirection, coord: T) -> miette::Result<T>
    where
        T: crate::IPjCoord,
    {
        let out_coord =
            unsafe { proj_sys::proj_trans(self.pj, i32::from(direction), coord.to_pj_coord()) };
        check_pj_result!(self);
        let out_coord = unsafe {
            T::from_pj_coord(
                out_coord.xyzt.x,
                out_coord.xyzt.y,
                out_coord.xyzt.z,
                out_coord.xyzt.t,
            )
        };
        Ok(out_coord)
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_get_last_used_operation>
    pub fn get_last_used_operation(&self) -> Self {
        unimplemented!()
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_generic>
    pub(crate) fn trans_generic(
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
        check_pj_result!(self);
        Ok(result)
    }
    /// Not suggested
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_array>
    pub fn trans_array<T>(
        &self,
        direction: crate::PjDirection,
        coord: &mut [T],
    ) -> miette::Result<&Self>
    where
        T: crate::IPjCoord,
    {
        let mut temp: Vec<crate::PjCoord> = coord.iter().map(|c| c.to_pj_coord()).collect();
        let code = unsafe {
            proj_sys::proj_trans_array(
                self.pj,
                i32::from(direction),
                coord.len(),
                temp.as_mut_ptr(),
            )
        };
        coord.iter_mut().zip(temp).for_each(|(c, t)| {
            *c = unsafe { T::from_pj_coord(t.xyzt.x, t.xyzt.y, t.xyzt.z, t.xyzt.t) }
        });
        check_pj_result!(self, code);
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
    #[test]
    fn test_trans() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let coord = [120.0, 30.0];
        let coord = pj.trans(crate::PjDirection::PjFwd, coord)?;
        assert_eq!(coord, [19955590.73888901, 3416780.562127255]);
        Ok(())
    }
    #[test]
    fn test_trans_array() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let mut coord = [[120.0, 30.0], [50.0, -80.0]];
        pj.trans_array(crate::PjDirection::PjFwd, &mut coord)?;
        assert_eq!(
            coord,
            [
                [19955590.73888901, 3416780.562127255],
                [17583572.872089125, -9356989.97994042]
            ]
        );
        Ok(())
    }
    #[test]
    fn test_trans_bounds() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
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
            crate::PjDirection::PjFwd,
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
        let ctx = crate::PjContext::default();
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
            crate::PjDirection::PjFwd,
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
