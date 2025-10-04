use crate::check_result;
use crate::data_types::ProjError;
// region:Coordinate transformation
impl crate::Proj {
    ///Return the operation used during the last invocation of
    /// [`Self::project`] or [`Self::convert`]. This is especially useful
    /// when P has been created with [`crate::Context::create_crs_to_crs()`]
    /// and has several alternative operations.
    ///
    ///  # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_get_last_used_operation>
    pub fn get_last_used_operation(&self) -> Option<crate::Proj> {
        use crate::Proj;

        let ptr = unsafe { proj_sys::proj_trans_get_last_used_operation(self.ptr()) };
        if ptr.is_null() {
            return None;
        }
        Some(Proj::new(&self.ctx, ptr).unwrap())
    }
    ///Transform a series of coordinates
    ///
    /// # Arguments
    ///
    /// * `direction`: Transformation direction.
    /// * `x`: Array of x-coordinates
    /// * `sx`: Step length, in bytes, between consecutive elements of the
    ///   corresponding array
    /// * `nx`: Number of elements in the corresponding array
    /// * `y`: Array of y-coordinates
    /// * `sy`: Step length, in bytes, between consecutive elements of the
    ///   corresponding array
    /// * `ny`: Number of elements in the corresponding array
    /// * `z`: Array of z-coordinates
    /// * `sz`: Step length, in bytes, between consecutive elements of the
    ///   corresponding array
    /// * `nz`: Number of elements in the corresponding array
    /// * `t`: Array of t-coordinates
    /// * `st`: Step length, in bytes, between consecutive elements of the
    ///   corresponding array
    /// * `nt`: Number of elements in the corresponding array
    ///
    /// # Safety
    ///
    /// If x,y is not null pointer.
    ///
    /// # See Also
    ///
    /// * [`Self::project`]
    /// * [`Self::convert`]
    /// * [`Self::project_array`]
    /// * [`Self::project_array`]
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_generic>
    pub unsafe fn trans_generic(
        &self,
        direction: crate::Direction,
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
    ) -> Result<usize, ProjError> {
        let result = unsafe {
            proj_sys::proj_trans_generic(
                self.ptr(),
                direction as i32,
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
}

impl crate::Context {
    ///Transform boundary.
    ///
    ///Transform boundary densifying the edges to account for nonlinear
    /// transformations along these edges and extracting the outermost bounds.
    ///
    ///If the destination CRS is geographic, the first axis is longitude, and
    /// *out_xmax < *out_xmin then the bounds crossed the antimeridian. In this
    /// scenario there are two polygons, one on each side of the antimeridian.
    /// The first polygon should be constructed with (*out_xmin, *out_ymin, 180,
    /// ymax) and the second with (-180, *out_ymin, *out_xmax, *out_ymax).
    ///
    ///If the destination CRS is geographic, the first axis is latitude, and
    /// *out_ymax < *out_ymin then the bounds crossed the antimeridian. In this
    /// scenario there are two polygons, one on each side of the antimeridian.
    /// The first polygon should be constructed with (*out_ymin, *out_xmin,
    /// *out_ymax, 180) and the second with (*out_ymin, -180, *out_ymax,
    /// *out_xmax).
    ///
    /// # Arguments
    ///
    /// * `P`: The PJ object representing the transformation.
    /// * `direction`: The direction of the transformation.
    /// * `xmin`: Minimum bounding coordinate of the first axis in source CRS
    ///   (target CRS if direction is inverse).
    /// * `ymin`: Minimum bounding coordinate of the second axis in source CRS.
    ///   (target CRS if direction is inverse).
    /// * `xmax`: Maximum bounding coordinate of the first axis in source CRS.
    ///   (target CRS if direction is inverse).
    /// * `ymax`: Maximum bounding coordinate of the second axis in source CRS.
    ///   (target CRS if direction is inverse).
    ///
    /// # Returns
    ///
    /// * (out_xmin, out_ymin, out_xmax, out_ymax): bounding coordinate target
    ///   CRS.(source CRS if direction is inverse).
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds>
    pub fn trans_bounds(
        &self,
        p: &crate::Proj,
        direction: crate::Direction,
        xmin: f64,
        ymin: f64,
        xmax: f64,
        ymax: f64,
        densify_pts: i32,
    ) -> Result<(f64, f64, f64, f64), ProjError> {
        let mut out_xmin = f64::default();
        let mut out_ymin = f64::default();
        let mut out_xmax = f64::default();
        let mut out_ymax = f64::default();
        let code = unsafe {
            proj_sys::proj_trans_bounds(
                self.ptr,
                p.ptr(),
                direction as i32,
                xmin,
                ymin,
                xmax,
                ymax,
                &mut out_xmin,
                &mut out_ymin,
                &mut out_xmax,
                &mut out_ymax,
                densify_pts,
            )
        };
        check_result!(code != 1, "Failures encountered.");
        Ok((out_xmin, out_ymin, out_xmax, out_ymax))
    }
    ///Transform boundary, taking into account 3D coordinates.
    ///
    ///Transform boundary densifying the edges to account for nonlinear
    /// transformations along these edges and extracting the outermost bounds.
    ///
    ///Note that the current implementation is not "perfect" when the source
    /// CRS is geocentric, the target CRS is geographic, and the input bounding
    /// box includes the center of the Earth, a pole or the antimeridian. In
    /// those circumstances, exact values of the latitude of longitude of
    /// discontinuity will not be returned.
    ///
    ///If one of the source or target CRS of the transformation is not 3D, the
    /// values of *out_zmin / *out_zmax may not be significant.
    ///
    ///For 2D or "2.5D" transformation (that is planar component is
    /// geographic/coordinates and 3D axis is elevation), the documentation of
    /// [`Self::trans_bounds()`] applies.
    ///
    /// # Arguments
    ///
    /// * `P`: The PJ object representing the transformation.
    /// * `direction`: The direction of the transformation.
    /// * `xmin`: Minimum bounding coordinate of the first axis in source CRS
    ///   (target CRS if direction is inverse).
    /// * `ymin`: Minimum bounding coordinate of the second axis in source CRS.
    ///   (target CRS if direction is inverse).
    /// * `zmin`: Minimum bounding coordinate of the third axis in source CRS.
    ///   (target CRS if direction is inverse).
    /// * `xmax`: Maximum bounding coordinate of the first axis in source CRS.
    ///   (target CRS if direction is inverse).
    /// * `ymax`: Maximum bounding coordinate of the second axis in source CRS.
    ///   (target CRS if direction is inverse).
    /// * `zmax`: Maximum bounding coordinate of the third axis in source CRS.
    ///   (target CRS if direction is inverse).
    ///
    /// # Returns
    ///
    /// * (out_xmin, out_ymin, out_zmin, out_xmax, out_ymax, out_zmax): bounding
    ///   coordinate target CRS.(source CRS if direction is inverse).
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds_3D>
    pub fn trans_bounds_3d(
        &self,
        p: &crate::Proj,
        direction: crate::Direction,
        xmin: f64,
        ymin: f64,
        zmin: f64,
        xmax: f64,
        ymax: f64,
        zmax: f64,
        densify_pts: i32,
    ) -> Result<(f64, f64, f64, f64, f64, f64), ProjError> {
        let mut out_xmin = f64::default();
        let mut out_ymin = f64::default();
        let mut out_zmin = f64::default();
        let mut out_xmax = f64::default();
        let mut out_ymax = f64::default();
        let mut out_zmax = f64::default();
        let code = unsafe {
            proj_sys::proj_trans_bounds_3D(
                self.ptr,
                p.ptr(),
                direction as i32,
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
                densify_pts,
            )
        };
        check_result!(code != 1, "Failures encountered.");
        Ok((out_xmin, out_ymin, out_zmin, out_xmax, out_ymax, out_zmax))
    }
}
#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_get_last_used_operation() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let _ = pj.convert(&(120.0, 30.0))?;
        let last_op = pj.get_last_used_operation();
        assert!(last_op.is_some());
        println!("{:?}", last_op.unwrap().info());
        Ok(())
    }
    #[test]
    fn test_get_last_used_operation_null() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        let last_op = pj.get_last_used_operation();
        assert!(last_op.is_none());
        Ok(())
    }

    #[test]
    fn test_trans_bounds() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let xmin = 0.0;
        let ymin = 1.0;
        let xmax = 20.0;
        let ymax = 30.0;

        let (out_xmin, out_ymin, out_xmax, out_ymax) =
            ctx.trans_bounds(&pj, crate::Direction::Fwd, xmin, ymin, xmax, ymax, 21)?;
        println!("{out_xmin},{out_ymin},{out_xmax},{out_ymax}");
        assert_approx_eq!(f64, out_xmin, 1799949.56320294);
        assert_approx_eq!(f64, out_ymin, 6639816.584496002);
        assert_approx_eq!(f64, out_xmax, 10788961.870597329);
        assert_approx_eq!(f64, out_ymax, 19555124.881683525);
        Ok(())
    }

    #[test]
    fn test_trans_bounds_3d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4496", &crate::Area::default())?;
        let pj = ctx.normalize_for_visualization(&pj)?;
        let xmin = 0.0;
        let ymin = 1.0;
        let zmin = 1.0;
        let xmax = 20.0;
        let ymax = 30.0;
        let zmax = 3.0;

        let (out_xmin, out_ymin, out_zmin, out_xmax, out_ymax, out_zmax) = ctx.trans_bounds_3d(
            &pj,
            crate::Direction::Fwd,
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
            21,
        )?;
        assert_approx_eq!(f64, out_xmin, 1799949.56320294);
        assert_approx_eq!(f64, out_ymin, 6639816.584496002);
        assert_approx_eq!(f64, out_zmin, 1.0);
        assert_approx_eq!(f64, out_xmax, 10788961.870597329);
        assert_approx_eq!(f64, out_ymax, 19555124.881683525);
        assert_approx_eq!(f64, out_zmax, 3.0);
        Ok(())
    }
}
