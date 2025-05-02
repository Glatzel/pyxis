use crate::check_pj_result;
impl crate::Pj {
    pub fn lp_dist(&self, a: crate::PjCoord, b: crate::PjCoord) -> miette::Result<f64> {
        let dist = unsafe { proj_sys::proj_lp_dist(self.pj, a, b) };
        check_pj_result!(self);
        Ok(dist)
    }
    pub fn lpz_dist(&self, a: crate::PjCoord, b: crate::PjCoord) -> miette::Result<f64> {
        let dist = unsafe { proj_sys::proj_lpz_dist(self.pj, a, b) };
        check_pj_result!(self);
        Ok(dist)
    }
    pub fn geod(&self, a: crate::PjCoord, b: crate::PjCoord) -> miette::Result<crate::PjCoord> {
        let dist = unsafe { proj_sys::proj_geod(self.pj, a, b) };
        check_pj_result!(self);
        Ok(dist)
    }
}

pub fn xy_dist(a: crate::PjCoord, b: crate::PjCoord) -> f64 {
    unsafe { proj_sys::proj_xy_dist(a, b) }
}

pub fn xyz_dist(a: crate::PjCoord, b: crate::PjCoord) -> f64 {
    unsafe { proj_sys::proj_xyz_dist(a, b) }
}
