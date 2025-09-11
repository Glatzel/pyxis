use std::sync::Arc;

use proj::Context;

pub fn init_proj_builder() ->Result<Arc<Context>,ProjError> {
    let ctx = proj::Context::new();
    ctx.set_log_level(proj::LogLevel::Trace)?;
    Ok(ctx)
}
