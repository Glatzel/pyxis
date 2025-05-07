use std::io::Write;
use std::path::PathBuf;

use miette::IntoDiagnostic;

const PROJ_DB: &[u8] = include_bytes!("proj.db");
pub fn init_proj_builder() -> miette::Result<proj::PjContext> {
    let ctx = proj::PjContext::new();

    // setup logging
    ctx.set_log_level(proj::PjLogLevel::Trace)?;

    // search for proj.db
    if let Ok(proj_data) = std::env::var("PROJ_DATA") {
        clerk::info!("PROJ_DATA environment variable is found: {proj_data}");
        ctx.set_search_paths(&[&PathBuf::from(proj_data)])?;
    } else {
        clerk::debug!("PROJ_DATA environment variable is not found");
        let exe_path = std::env::current_exe().into_diagnostic()?;
        let exe_root = exe_path.parent().unwrap();
        if !exe_root.join("proj.db").exists() {
            clerk::warn!("proj.db is not found. Try to use bundled proj.db");
            let mut db_file = std::fs::File::create(exe_root.join("proj.db")).into_diagnostic()?;
            db_file.write_all(PROJ_DB).into_diagnostic()?;
            clerk::info!("Write to: {}", exe_root.join("proj.db").to_str().unwrap());
        }
        ctx.set_search_paths(&[exe_root])?;
    }
    Ok(ctx)
}
