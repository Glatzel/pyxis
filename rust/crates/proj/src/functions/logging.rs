use std::ffi::c_void;

use miette::IntoDiagnostic;

use crate::check_result;
use crate::data_types::LogLevel;
/// # Logging
impl crate::Context {
    /// See [`Self::set_log_level`]
    ///
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_log_level>
    pub fn log_level(&self, level: LogLevel) -> miette::Result<LogLevel> {
        let level = unsafe { proj_sys::proj_log_level(self.ptr(), level.into()) };
        let level = LogLevel::try_from(level).into_diagnostic()?;
        Ok(level)
    }
    /// See [`Self::set_log_func`]
    ///
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_log_func>
    #[deprecated(since = "0.0.21", note = "Use `Self::set_log_level` instead.")]
    fn _log_func(
        &self,
        app_data: *mut c_void,
        logf: Option<unsafe extern "C" fn(*mut c_void, i32, *const i8)>,
    ) -> miette::Result<&Self> {
        //initialize log
        unsafe {
            proj_sys::proj_log_func(self.ptr(), app_data, logf);
        };
        check_result!(self);
        Ok(self)
    }
}
