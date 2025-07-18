use std::ffi::c_void;
use std::sync::Arc;

use envoy::CStrToString;
use miette::IntoDiagnostic;

use crate::{LogLevel, check_result};

pub(crate) unsafe extern "C" fn proj_clerk(_: *mut c_void, level: i32, info: *const i8) {
    let _message = info.to_string().unwrap_or_default();

    match level {
        1 => clerk::error!("{}", _message),
        2 => clerk::debug!("{}", _message),
        3 => clerk::trace!("{}", _message),
        _ => (),
    };
}

impl crate::Context {
    pub fn set_log_level(self: Arc<Self>, level: LogLevel) -> miette::Result<Arc<Self>> {
        let level = unsafe { proj_sys::proj_log_level(self.ptr, level.into()) };
        let _ = LogLevel::try_from(level).into_diagnostic()?;
        Ok(self)
    }

    pub(crate) fn set_log_fn(
        self: Arc<Self>,
        app_data: *mut c_void,
        logf: Option<unsafe extern "C" fn(*mut c_void, i32, *const i8)>,
    ) -> miette::Result<Arc<Self>> {
        unsafe {
            proj_sys::proj_log_func(self.ptr, app_data, logf);
        };
        check_result!(self);
        Ok(self)
    }
}

#[cfg(test)]
mod test {
    use crate::LogLevel;

    #[test]
    fn test_log() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.clone().set_log_level(LogLevel::Trace)?;
        let _ = ctx.create("EPSG:4326")?;

        Ok(())
    }

    #[test]
    fn test_log_error() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.clone().set_log_level(LogLevel::Trace)?;
        let pj = ctx.create("Unknown crs");
        assert!(pj.is_err());
        Ok(())
    }

    #[test]
    fn test_log_change_level() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.clone().set_log_level(LogLevel::Debug)?;
        let pj = ctx.clone().create("Show log");
        assert!(pj.is_err());
        ctx.clone().set_log_level(LogLevel::None)?;
        let pj = ctx.create("Hide log");
        assert!(pj.is_err());
        Ok(())
    }
}
