use std::env;
extern crate alloc;

use crate::Context;
use crate::data_types::ProjError;

pub(crate) fn new_test_ctx() -> Result<Context, ProjError> {
    clerk::init_log_with_level(clerk::LogLevel::TRACE);
    let ctx = crate::Context::new();
    ctx.set_log_level(crate::LogLevel::Trace)?;
    ctx.set_enable_network(true)?;
    // PROJ_DATA
    let workspace_root = env::var("CARGO_WORKSPACE_DIR").unwrap();
    let default_proj_data = if cfg!(target_os = "windows") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/Library/share/proj"
        ))?
    } else {
        dunce::canonicalize(format!("{workspace_root}/.pixi/envs/default/share/proj"))?
    };
    ctx.set_database_path(&default_proj_data.join("proj.db"), None)?;
    ctx.set_search_paths(&[&default_proj_data])?;
    Ok(ctx)
}
