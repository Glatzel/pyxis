use std::env;
use std::sync::Arc;

use miette::IntoDiagnostic;

use crate::Context;

pub(crate) fn new_test_ctx() -> miette::Result<Arc<Context>> {
    clerk::init_log_with_level(clerk::LogLevel::TRACE);
    let ctx = crate::Context::new();
    ctx.set_log_level(crate::LogLevel::Trace)?;
    ctx.set_enable_network(true)?;
    Ok(ctx)
}
