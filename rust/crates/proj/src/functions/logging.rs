use miette::IntoDiagnostic;

use crate::LogLevel;
/// # Logging
impl crate::Context {
    /// # See Also
    ///
    /// * [`Self::set_log_level`]
    ///
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_log_level>
    pub fn log_level(&self, level: LogLevel) -> miette::Result<LogLevel> {
        let level = unsafe { proj_sys::proj_log_level(self.ptr, level.into()) };
        let level = LogLevel::try_from(level).into_diagnostic()?;
        Ok(level)
    }
    /// # See Also
    ///
    /// * [`Self::set_log_fn`]
    ///
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_log_func>
    #[deprecated(since = "0.0.21", note = "Use `Self::set_log_level` instead.")]
    fn _log_func(&self) -> miette::Result<&Self> { unimplemented!() }
}
