use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::data_types::PjLogLevel;

pub(crate) fn new_test_ctx() -> miette::Result<crate::PjContext> {
    tracing_subscriber::registry()
        .with(clerk::terminal_layer(LevelFilter::TRACE, true))
        .init();
    let ctx = crate::PjContext::default();
    ctx.set_log_level(PjLogLevel::Trace)?;
    Ok(ctx)
}
