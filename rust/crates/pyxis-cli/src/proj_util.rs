use std::io::Write;
use std::path::PathBuf;

use miette::IntoDiagnostic;

const PROJ_DB: &[u8] = include_bytes!("proj.db");

pub fn init_proj_builder() -> miette::Result<proj::PjContext> {
    let ctx = proj::PjContext::new();
    if let Ok(proj_data) = std::env::var("PROJ_DATA") {
        clerk::info!("PROJ_DATA environment variable is found: {proj_data}");
        ctx.set_search_paths(&[&PathBuf::from(proj_data)])?;
    } else {
        clerk::info!("PROJ_DATA environment variable is not found");
        let exe_path = std::env::current_exe().unwrap();
        let exe_root = exe_path.parent().unwrap();
        if !exe_root.join("proj.db").exists() {
            clerk::warn!("proj.db is not found.");

            {
                clerk::info!("Write to: {}", exe_root.join("proj.db").to_str().unwrap());
                let mut db_file = std::fs::File::create(exe_root.join("proj.db")).unwrap();
                db_file.write_all(PROJ_DB).unwrap();
            }
        }
        ctx.set_search_paths(&[&std::env::current_exe().into_diagnostic()?.parent().unwrap()])?;
    }
    Ok(ctx)
}
