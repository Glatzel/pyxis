use std::env;

use clerk::LevelFilter;
use miette::IntoDiagnostic;

use crate::LogLevel;

pub(crate) fn new_test_ctx() -> miette::Result<crate::Context> {
    clerk::init_log_with_level(LevelFilter::TRACE);
    let ctx = crate::Context::default();
    ctx.set_log_level(LogLevel::Trace)?;
    ctx.set_enable_network(true)?;
    // PROJ_DATA
    let workspace_root = env::var("CARGO_WORKSPACE_DIR").unwrap();
    let default_proj_data = if cfg!(target_os = "windows") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/x64-windows-static/share/proj"
        ))
        .into_diagnostic()?
    } else if cfg!(target_os = "linux") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/x64-linux-release/share/proj"
        ))
        .into_diagnostic()?
    } else if cfg!(target_os = "macos") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/arm64-osx-release/share/proj"
        ))
        .into_diagnostic()?
    } else {
        panic!("Unsupported OS")
    };
    ctx.set_database_path(&default_proj_data.join("proj.db"), None)?;
    ctx.set_search_paths(&[&default_proj_data])?;
    Ok(ctx)
}
