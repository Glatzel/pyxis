use crate::{check_context_result, check_pj_result};

// region:Coordinate transformation
/// # References
///<https://proj.org/en/stable/development/reference/functions.html#coordinate-transformation>
impl crate::Pj {
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans>
    pub fn trans(
        &self,
        direction: crate::PjDirection,
        coord: crate::PjCoord,
    ) -> miette::Result<crate::PjCoord> {
        let out_coord = unsafe { proj_sys::proj_trans(self.pj, i32::from(direction), coord) };
        check_pj_result!(self);
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
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_array>
    pub fn trans_array(
        &self,
        direction: crate::PjDirection,
        coord: &mut [crate::PjCoord],
    ) -> miette::Result<&Self> {
        let code = unsafe {
            proj_sys::proj_trans_array(
                self.pj,
                i32::from(direction),
                coord.len(),
                coord.as_mut_ptr(),
            )
        };
        check_pj_result!(self, code);
        Ok(self)
    }
}
impl crate::PjContext {
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds>
    pub fn _trans_bounds(
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
        check_context_result!(self, code);
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
        check_context_result!(self, code);
        Ok(self)
    }
}
