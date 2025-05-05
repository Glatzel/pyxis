///# Error reporting
impl crate::Pj {
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno>
    pub(crate) fn errno(&self) -> crate::PjErrorCode {
        crate::PjErrorCode::from(unsafe { proj_sys::proj_errno(self.pj) } as u32)
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_set>
    pub(crate) fn _errno_set(&self, err: &crate::PjErrorCode) -> &Self {
        unsafe { proj_sys::proj_errno_set(self.pj, i32::from(err)) };
        self
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_reset>
    pub(crate) fn _errno_reset(&self) -> crate::PjErrorCode {
        crate::PjErrorCode::from(unsafe { proj_sys::proj_errno_reset(self.pj) } as u32)
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_restore>
    pub(crate) fn _errno_restore(&self, err: &crate::PjErrorCode) -> &Self {
        unsafe { proj_sys::proj_errno_restore(self.pj, i32::from(err)) };
        self
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string>
    pub(crate) fn errno_string(&self, err: &crate::PjErrorCode) -> String {
        crate::c_char_to_string(unsafe { proj_sys::proj_errno_string(i32::from(err)) })
    }
}
///# Error reporting
impl crate::PjContext {
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno>
    pub(crate) fn errno(&self) -> crate::PjErrorCode {
        crate::PjErrorCode::from(unsafe { proj_sys::proj_context_errno(self.ctx) } as u32)
    }
    pub(crate) fn errno_string(&self, err: &crate::PjErrorCode) -> String {
        crate::c_char_to_string(unsafe {
            proj_sys::proj_context_errno_string(self.ctx, i32::from(err))
        })
    }
}
