use envoy::CStrToString;

use crate::data_types::ProjError;
///# Error reporting
impl crate::Proj {
    /// Get a reading of the current error-state of P. An non-zero error codes
    /// indicates an error either with the transformation setup or during a
    /// transformation. In cases P is 0 the error number of the default context
    /// is read. A text representation of the error number can be retrieved with
    /// [`Self::errno_string()`].
    ///
    ///  # See Also
    ///
    /// *[`crate::check_result`]
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_errno>
    pub(crate) fn errno(&self) -> ProjError {
        ProjError::from(unsafe { proj_sys::proj_errno(self.ptr()) })
    }
    ///Change the error-state of Proj to err.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_set>
    pub(crate) fn _errno_set(&self, err: ProjError) -> &Self {
        unsafe { proj_sys::proj_errno_set(self.ptr(), err.into()) };
        self
    }
    ///Clears the error number in P, and bubbles it up to the context.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_reset>
    pub(crate) fn _errno_reset(&self) -> ProjError {
        ProjError::from(unsafe { proj_sys::proj_errno_reset(self.ptr()) })
    }
    ///Reduce some mental impedance in the canonical reset/restore use case:
    /// Basically, [`Self::_errno_restore()`] is a synonym for
    /// [`Self::_errno_set()`], but the use cases are very different: set
    /// indicate an error to higher level user code, restore passes
    /// previously set error indicators in case of no errors at this level.
    ///
    ///Hence, although the inner working is identical, we provide both options,
    /// to avoid some rather confusing real world code.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_restore>
    pub(crate) fn _errno_restore(&self, err: ProjError) -> &Self {
        unsafe { proj_sys::proj_errno_restore(self.ptr(), err.into()) };
        self
    }
    ///Get a text representation of an error number.
    ///
    /// Since the original function is potentially thread-unsafe, use
    /// [`crate::Context::errno_string`] here.
    ///
    /// # See Also
    ///
    /// * [`crate::check_result`]
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string>
    pub(crate) fn errno_string(&self, err: ProjError) -> String { self.ctx.errno_string(err) }
}

///# Error reporting
impl crate::Context {
    /// Get a reading of the current error-state of P. An non-zero error codes
    /// indicates an error either with the transformation setup or during a
    /// transformation. In cases P is 0 the error number of the default context
    /// is read. A text representation of the error number can be retrieved with
    /// [`Self::errno_string()`].
    ///
    /// # See Also
    ///
    /// * [`crate::check_result`]
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno>
    pub(crate) fn errno(&self) -> ProjError {
        ProjError::from(unsafe { proj_sys::proj_context_errno(self.ptr) })
    }

    ///Get a text representation of an error number.
    ///
    /// # See Also
    ///
    /// * [`crate::check_result`]
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string>
    pub(crate) fn errno_string(&self, err: ProjError) -> String {
        unsafe { proj_sys::proj_context_errno_string(self.ptr, err.into()) }
            .to_string()
            .unwrap_or("Unknown error.".to_string())
    }
}
