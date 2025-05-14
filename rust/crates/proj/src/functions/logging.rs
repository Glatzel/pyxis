use std::ffi::c_void;

use crate::check_result;
use crate::data_types::PjLogLevel;
/// # Logging
impl crate::PjContext {
    /// See [`Self::set_log_level`]
    ///
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_log_level>
    pub fn log_level(&self, level: PjLogLevel) -> miette::Result<PjLogLevel> {
        let level = unsafe { proj_sys::proj_log_level(self.ctx, level.into()) };
        let level = PjLogLevel::try_from(level)?;
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
            proj_sys::proj_log_func(self.ctx, app_data, logf);
        };
        check_result!(self);
        Ok(self)
    }
}
