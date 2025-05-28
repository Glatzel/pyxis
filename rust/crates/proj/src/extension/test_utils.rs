use std::env;
use std::path::PathBuf;

use miette::IntoDiagnostic;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::data_types::LogLevel;

pub(crate) fn new_test_ctx() -> miette::Result<crate::Context> {
    tracing_subscriber::registry()
        .with(clerk::terminal_layer(LevelFilter::TRACE, true))
        .init();
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
        .to_string_lossy()
        .to_string()
    } else if cfg!(target_os = "linux") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/x64-linux-release/share/proj"
        ))
        .into_diagnostic()?
        .to_string_lossy()
        .to_string()
    } else if cfg!(target_os = "macos") {
        dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/arm64-osx-release/share/proj"
        ))
        .into_diagnostic()?
        .to_string_lossy()
        .to_string()
    } else {
        panic!("Unsupported OS")
    };
    ctx.set_search_paths(&[&PathBuf::from(default_proj_data)])?;
    Ok(ctx)
}
