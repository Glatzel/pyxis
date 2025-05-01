use super::error_codes::PjErrorCode;
pub struct Pj {
    pub(crate) pj: *mut proj_sys::PJ,
}
impl Drop for Pj {
    fn drop(&mut self) {
        unsafe { proj_sys::proj_destroy(self.pj) };
    }
}
// region:Coordinate transformation
/// #References
///<https://proj.org/en/stable/development/reference/functions.html#coordinate-transformation>
impl Pj {
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans>
    pub fn trans(&self) {
        unimplemented!()
    }
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_get_last_used_operation>
    pub fn trans_get_last_used_operation(&self) {
        unimplemented!()
    }
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_generic>
    pub fn proj_trans_generic(&self) {
        unimplemented!()
    }
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_array>
    pub fn proj_trans_array(&self) {
        unimplemented!()
    }
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds>
    pub fn proj_trans_bounds(&self) {
        unimplemented!()
    }
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds_3D>
    pub fn proj_trans_bounds_3d(&self) {
        unimplemented!()
    }
}
impl Pj {
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno>
    fn _errno(&self) -> PjErrorCode {
        PjErrorCode::from(unsafe { proj_sys::proj_errno(self.pj) } as u32)
    }
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_set>
    fn _errno_set(&self, err: PjErrorCode) -> &Self {
        unsafe { proj_sys::proj_errno_set(self.pj, i32::from(err)) };
        self
    }
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_reset>
    fn _errno_reset(&self) -> PjErrorCode {
        PjErrorCode::from(unsafe { proj_sys::proj_errno_reset(self.pj) } as u32)
    }
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_restore>
    fn _errno_restore(&self, err: PjErrorCode) -> &Self {
        unsafe { proj_sys::proj_errno_restore(self.pj, i32::from(err)) };
        self
    }
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string>
    fn _errno_string(&self, err: PjErrorCode) -> String {
        crate::c_char_to_string(unsafe { proj_sys::proj_errno_string(i32::from(err)) })
    }
}
/// #References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_DIRECTION>
pub enum PjDirection {
    PjFwd,
    PjIdent,
    PjInv,
}

pub struct PjContext {
    pub(crate) ctx: *mut proj_sys::PJ_CONTEXT,
}

impl PjContext {
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno>
    fn _errno(&self) -> PjErrorCode {
        PjErrorCode::from(unsafe { proj_sys::proj_context_errno(self.ctx) } as u32)
    }
    fn _errno_string(&self, err: PjErrorCode) -> String {
        crate::c_char_to_string(unsafe {
            proj_sys::proj_context_errno_string(self.ctx, i32::from(err))
        })
    }
}
pub struct PjArea {
    pub(crate) area: *mut proj_sys::PJ_AREA,
}
