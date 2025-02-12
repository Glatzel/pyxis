use std::fmt;
use std::sync::LazyLock;

use console::Style;
use tracing::{Event, Subscriber};
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt::format::{FormatEvent, FormatFields};
use tracing_subscriber::fmt::{format, FmtContext};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::EnvFilter;

// console style
pub static TRACE_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().color256(99));
pub static DEBUG_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().blue());
pub static INFO_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().green());
pub static WARN_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().yellow().bold());
pub static ERROR_STYLE: LazyLock<Style> = LazyLock::new(|| Style::new().red().bold());

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
    use tracing::{debug, error, info, trace, warn};

    use super::*;
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
