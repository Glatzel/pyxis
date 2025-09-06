use std::env;
extern crate alloc;
use alloc::sync::Arc;

use crate::Context;

pub(crate) fn new_test_ctx() -> mischief::Result<Arc<Context>> {
    clerk::init_log_with_level(clerk::LogLevel::TRACE);
    let ctx = crate::Context::new();
    ctx.set_log_level(crate::LogLevel::Trace)?;
    ctx.set_enable_network(true)?;
    // PROJ_DATA
    let workspace_root = env::var("CARGO_WORKSPACE_DIR").unwrap();
    let default_proj_data = if cfg!(target_os = "windows") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/x64-windows-static/share/proj"
        ))?
    } else if cfg!(target_os = "macos") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/arm64-osx-release/share/proj"
        ))?
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/x64-linux-release/share/proj"
        ))?
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/arm64-linux-release/share/proj"
        ))?
    } else {
        panic!("Unsupported OS")
    };
    ctx.set_database_path(&default_proj_data.join("proj.db"), None)?;
    ctx.set_search_paths(&[&default_proj_data])?;
    Ok(ctx)
}
