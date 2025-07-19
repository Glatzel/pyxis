use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use miette::IntoDiagnostic;
use proj::Context;

// const PROJ_DB: &[u8] = include_bytes!(concat!(env!("PROJ_DATA"), "/proj.db"));
pub fn init_proj_builder() -> miette::Result<Arc<Context>> {
    let ctx = proj::Context::new();
    ctx.set_log_level(proj::LogLevel::Trace)?;
    Ok(ctx)
}
