use std::path::PathBuf;

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
    // PROJ_DATA
    let default_proj_data = match env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "windows" => {
            dunce::canonicalize("../../.pixi/envs/default/proj/x64-windows-static/share/proj")
                .unwrap()
                .to_string_lossy()
                .to_string()
        }
        "linux" => {
            dunce::canonicalize("../../.pixi/envs/default/proj/x64-linux-release/share/proj")
                .unwrap()
                .to_string_lossy()
                .to_string()
        }
        "macos" => {
            dunce::canonicalize("../../.pixi/envs/default/proj/arm64-osx-release/share/proj")
                .unwrap()
                .to_string_lossy()
                .to_string()
        }
        other => {
            panic!("Unsupported OS: {}", other)
        }
    };
    ctx.set_search_paths(&[&PathBuf::from(default_proj_data)]);
    Ok(ctx)
}
