use super::constants::{DEBUG_STYLE, ERROR_STYLE, INFO_STYLE, TRACE_STYLE, WARN_STYLE};
use std::fmt;
use tracing::{Event, Subscriber};
use tracing_subscriber::fmt::format;
use tracing_subscriber::fmt::{
    format::{FormatEvent, FormatFields},
    FmtContext,
};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::{filter::LevelFilter, EnvFilter};
pub fn init_logger(level: LevelFilter) {
    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(level.into())
                .from_env_lossy(),
        )
        .with_writer(std::io::stderr)
        .event_format(VinayaLogFormatter)
        .finish();
    tracing::subscriber::set_global_default(subscriber).unwrap();
}
struct VinayaLogFormatter;

fn color_level(level: &tracing::Level) -> console::StyledObject<std::string::String> {
    match *level {
        tracing::Level::TRACE => TRACE_STYLE.apply_to(level.to_string()),
        tracing::Level::DEBUG => DEBUG_STYLE.apply_to(level.to_string()),
        tracing::Level::INFO => INFO_STYLE.apply_to(level.to_string()),
        tracing::Level::WARN => WARN_STYLE.apply_to(level.to_string()),
        tracing::Level::ERROR => ERROR_STYLE.apply_to(level.to_string()),
    }
}

impl<S, N> FormatEvent<S, N> for VinayaLogFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &Event<'_>,
    ) -> fmt::Result {
        write!(
            writer,
            "[{}] [{:}] [{}] [{}:{}] ",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"), // Custom timestamp format
            color_level(event.metadata().level()),
            event.metadata().target(),
            event.metadata().file().unwrap_or("<file>"),
            event.metadata().line().unwrap_or(0),
        )?;

        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tracing::{debug, error, info, trace, warn};
    #[test]
    fn test_log() {
        init_logger(LevelFilter::TRACE);
        trace!("Trace message");
        debug!("Debug message");
        info!("Informational message");
        warn!("Warning message");
        error!("Error message");
    }
}
