use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
pub(crate) fn init_test_ctx() -> crate::PjContext {
    tracing_subscriber::registry()
        .with(clerk::terminal_layer(LevelFilter::TRACE, true))
        .init();
    let ctx = crate::PjContext::default();
    ctx.set_log_level(crate::PjLogLevel::Trace);
    ctx
}
