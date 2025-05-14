use crate::data_types::PjError;

///# Error reporting
impl crate::Pj<'_> {
    /// See [`crate::check_result`]
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno>
    pub(crate) fn errno(&self) -> PjError {
        PjError::from(unsafe { proj_sys::proj_errno(self.ptr) })
    }

    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_set>
    pub(crate) fn _errno_set(&self, err: &PjError) -> &Self {
        unsafe { proj_sys::proj_errno_set(self.ptr, i32::from(err)) };
        self
    }

    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_reset>
    pub(crate) fn _errno_reset(&self) -> PjError {
        PjError::from(unsafe { proj_sys::proj_errno_reset(self.ptr) })
    }

    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_restore>
    pub(crate) fn _errno_restore(&self, err: &PjError) -> &Self {
        unsafe { proj_sys::proj_errno_restore(self.ptr, i32::from(err)) };
        self
    }

    /// See [`crate::check_result`]
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string>
    pub(crate) fn errno_string(&self, err: &PjError) -> String {
        crate::c_char_to_string(unsafe { proj_sys::proj_errno_string(i32::from(err)) })
            .unwrap_or("Unknown error.".to_string())
    }
}

///# Error reporting
impl crate::PjContext {
    /// See [`crate::check_result`]
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno>
    pub(crate) fn errno(&self) -> PjError {
        PjError::from(unsafe { proj_sys::proj_context_errno(self.ptr) })
    }

    /// See [`crate::check_result`]
    /// ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string>
    pub(crate) fn errno_string(&self, err: &PjError) -> String {
        crate::c_char_to_string(unsafe {
            proj_sys::proj_context_errno_string(self.ptr, i32::from(err))
        })
        .unwrap_or("Unknown error.".to_string())
    }
}
