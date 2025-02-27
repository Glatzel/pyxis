#[cfg(target_os = "windows")]
use std::io::Write;
use std::path::PathBuf;
#[cfg(target_os = "windows")]
const PROJ_DB: &[u8] = include_bytes!("proj.db");

pub fn init_proj_builder() -> proj::ProjBuilder {
    let mut builder = proj::ProjBuilder::new();
    if let Ok(proj_data) = std::env::var("PROJ_DATA") {
        tracing::info!("PROJ_DATA environment variable is found: {proj_data}");
        builder.set_search_paths(PathBuf::from(proj_data)).unwrap();
    } else {
        tracing::info!("PROJ_DATA environment variable is not found");
        let exe_root = std::env::current_exe().unwrap();
        exe_root.parent().unwrap();
        if !exe_root.clone().join("proj.db").exists() {
            tracing::info!("proj.db is not found. Use bundled file.");
            #[cfg(target_os = "windows")]
            {
                let mut db_file = std::fs::File::create("proj.db").unwrap();
                db_file.write_all(PROJ_DB).unwrap();
            }
        }
        builder
            .set_search_paths(std::env::current_exe().unwrap().parent().unwrap())
            .unwrap();
    }
    builder
}
