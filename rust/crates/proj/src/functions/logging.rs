use core::ffi::{c_char, c_void};

use crate::LogLevel;
use crate::data_types::ProjError;
use crate::error_handling::check_result;

impl crate::Context {
    pub fn log_level(&self, level: LogLevel) -> LogLevel {
        LogLevel::from(unsafe { proj_sys::proj_log_level(self.ptr(), level as u32) })
    }

    ///Override the internal log function of PROJ.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `app_data` remains valid and properly
    /// aligned for as long as PROJ may invoke `logf`. The caller must also
    /// ensure that `logf`, when provided, is a valid `unsafe extern "C"`
    /// function pointer and is safe to call with the supplied `app_data`.
    ///
    /// The callback must not outlive the data referenced by `app_data`.
    ///
    /// # Reference
    ///
    /// * <https://proj.org/development/reference/functions.html#proj_log_func>
    pub unsafe fn log_func<T>(
        &self,
        app_data: &mut T,
        logf: Option<unsafe extern "C" fn(*mut c_void, i32, *const c_char)>,
    ) -> Result<&Self, ProjError> {
        unsafe {
            proj_sys::proj_log_func(self.ptr(), app_data as *mut T as *mut c_void, logf);
        };
        check_result!(self);
        Ok(self)
    }
}

#[cfg(test)]
mod test {
    use crate::LogLevel;

    #[test]
    fn test_log() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.log_level(LogLevel::Trace);
        let _ = ctx.create("EPSG:4326")?;

        Ok(())
    }

    #[test]
    fn test_log_error() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.log_level(LogLevel::Trace);
        let pj = ctx.create("Unknown crs");
        assert!(pj.is_err());
        Ok(())
    }

    #[test]
    fn test_log_change_level() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.log_level(LogLevel::Debug);
        let pj = ctx.create("Show log");
        assert!(pj.is_err());
        ctx.log_level(LogLevel::None);
        let pj = ctx.create("Hide log");
        assert!(pj.is_err());
        Ok(())
    }
}
