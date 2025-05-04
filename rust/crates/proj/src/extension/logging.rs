use std::ffi::c_void;

use crate::PjLogLevel;
pub(crate) unsafe extern "C" fn proj_clerk(_: *mut c_void, level: i32, info: *const i8) {
    let message = crate::c_char_to_string(info);

    match level {
        1 => clerk::error!("{}", message),
        2 => clerk::debug!("{}", message),
        3 => clerk::trace!("{}", message),
        _ => (),
    };
}
impl crate::PjContext {
    pub fn set_log_level(&self, level: PjLogLevel) -> &Self {
        unsafe {
            proj_sys::proj_log_level(self.ctx, i32::from(level));
        };
        self
    }
}
#[cfg(test)]
mod test {
    use tracing::level_filters::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    #[test]
    fn test_log() -> miette::Result<()> {
        tracing_subscriber::registry()
            .with(clerk::terminal_layer(LevelFilter::TRACE, true))
            .init();
        let ctx = crate::PjContext::default();
        ctx.set_log_level(crate::PjLogLevel::Trace);
        let _ = ctx.create("EPSG:4326")?;

        Ok(())
    }
    #[test]
    fn test_log_error() -> miette::Result<()> {
        tracing_subscriber::registry()
            .with(clerk::terminal_layer(LevelFilter::DEBUG, true))
            .init();
        let ctx = crate::PjContext::default();
        ctx.set_log_level(crate::PjLogLevel::Trace);
        let pj = ctx.create("unknow crs");
        assert!(pj.is_err());
        Ok(())
    }
    #[test]
    fn test_log_change_level() -> miette::Result<()> {
        tracing_subscriber::registry()
            .with(clerk::terminal_layer(LevelFilter::TRACE, true))
            .init();
        let ctx = crate::PjContext::default();
        ctx.set_log_level(crate::PjLogLevel::Debug);
        let pj = ctx.create("Show log");
        assert!(pj.is_err());
        ctx.set_log_level(crate::PjLogLevel::None);
        let pj = ctx.create("Hide log");
        assert!(pj.is_err());
        Ok(())
    }
}
