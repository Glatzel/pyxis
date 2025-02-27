use std::path::PathBuf;
#[cfg(target_os = "windows")]
const _: &[u8] = include_bytes!("proj.db");

pub fn init_proj_builder() -> proj::ProjBuilder {
    let mut builder = proj::ProjBuilder::new();
    match std::env::var("PROJ_DATA") {
        Ok(proj_data) => {
            builder.set_search_paths(PathBuf::from(proj_data)).unwrap();
        }
        Err(_) => {}
    }

    builder
}
