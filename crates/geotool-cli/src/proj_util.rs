use std::path::PathBuf;
#[cfg(target_os = "windows")]
const _: &[u8] = include_bytes!("proj.db");

pub fn init_proj_builder() -> proj::ProjBuilder {
    let mut builder = proj::ProjBuilder::new();
    if let Ok(proj_data) = std::env::var("PROJ_DATA") {
        builder.set_search_paths(PathBuf::from(proj_data)).unwrap();
    }

    builder
}
