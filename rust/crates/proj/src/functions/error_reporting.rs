use crate::CstrToString;
use crate::data_types::ProjError;
///# Error reporting
impl crate::Proj<'_> {
    /// # See Also
    ///
    /// *[`crate::check_result`]
    ///
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_errno>
    pub(crate) fn errno(&self) -> ProjError {
        ProjError::from(unsafe { proj_sys::proj_errno(self.ptr()) })
    }

    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_set>
    pub(crate) fn _errno_set(&self, err: ProjError) -> &Self {
        unsafe { proj_sys::proj_errno_set(self.ptr(), err.into()) };
        self
    }

    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_reset>
    pub(crate) fn _errno_reset(&self) -> ProjError {
        ProjError::from(unsafe { proj_sys::proj_errno_reset(self.ptr()) })
    }

    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_restore>
    pub(crate) fn _errno_restore(&self, err: ProjError) -> &Self {
        unsafe { proj_sys::proj_errno_restore(self.ptr(), err.into()) };
        self
    }

    /// # See Also
    ///
    /// * [`crate::check_result`]
    ///
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string>
    pub(crate) fn errno_string(&self, err: ProjError) -> String {
        unsafe { proj_sys::proj_errno_string(err.into()) }
            .to_string()
            .unwrap_or("Unknown error.".to_string())
    }
}

///# Error reporting
impl crate::Context {
    /// # See Also
    ///
    /// * [`crate::check_result`]
    ///
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno>
    pub(crate) fn errno(&self) -> ProjError {
        ProjError::from(unsafe { proj_sys::proj_context_errno(self.ptr) })
    }

    /// # See Also
    ///
    /// * [`crate::check_result`]
    ///
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string>
    pub(crate) fn errno_string(&self, err: ProjError) -> String {
        unsafe { proj_sys::proj_context_errno_string(self.ptr, err.into()) }
            .to_string()
            .unwrap_or("Unknown error.".to_string())
    }
}
