use std::ffi::c_void;

use crate::data_types::PjLogLevel;

pub(crate) unsafe extern "C" fn proj_clerk(_: *mut c_void, level: i32, info: *const i8) {
    let _message = crate::c_char_to_string(info);

    match level {
        1 => {
            clerk::error!("{}", _message);
        }
        2 => {
            clerk::debug!("{}", _message);
        }
        3 => {
            clerk::trace!("{}", _message);
        }
        _ => (),
    };
}

impl crate::PjContext {
    pub fn set_log_level(&self, level: PjLogLevel) -> miette::Result<&Self> {
        self.log_level(level)?;
        Ok(self)
    }
    pub fn set_log_func(
        &self,
        app_data: *mut c_void,
        logf: Option<unsafe extern "C" fn(*mut c_void, i32, *const i8)>,
    ) -> miette::Result<&Self> {
        self.log_func(app_data, logf)
    }
}

#[cfg(test)]
mod test {
    use tracing::level_filters::LevelFilter;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    use crate::data_types::PjLogLevel;

    #[test]
    fn test_log() -> miette::Result<()> {
        tracing_subscriber::registry()
            .with(clerk::terminal_layer(LevelFilter::TRACE, true))
            .init();
        let ctx = crate::PjContext::default();
        ctx.set_log_level(PjLogLevel::Trace)?;
        let _ = ctx.create("EPSG:4326")?;

        Ok(())
    }

    #[test]
    fn test_log_error() -> miette::Result<()> {
        tracing_subscriber::registry()
            .with(clerk::terminal_layer(LevelFilter::DEBUG, true))
            .init();
        let ctx = crate::PjContext::default();
        ctx.set_log_level(PjLogLevel::Trace)?;
        let pj = ctx.create("Unknown crs");
        assert!(pj.is_err());
        Ok(())
    }

    #[test]
    fn test_log_change_level() -> miette::Result<()> {
        tracing_subscriber::registry()
            .with(clerk::terminal_layer(LevelFilter::TRACE, true))
            .init();
        let ctx = crate::PjContext::default();
        ctx.set_log_level(PjLogLevel::Debug)?;
        let pj = ctx.create("Show log");
        assert!(pj.is_err());
        ctx.set_log_level(PjLogLevel::None)?;
        let pj = ctx.create("Hide log");
        assert!(pj.is_err());
        Ok(())
    }
}
