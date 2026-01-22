use proj::Context;

pub fn init_proj_builder() -> mischief::Result<Context> {
    let ctx = proj::Context::new();
    ctx.set_log_level(proj::LogLevel::Trace)?;
    Ok(ctx)
}
