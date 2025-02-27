use std::io::Write;
use std::path::PathBuf;
#[cfg(target_os = "windows")]
const PROJ_DB: &[u8] = include_bytes!("proj.db");

pub fn init_proj_builder() -> proj::ProjBuilder {
    let mut builder = proj::ProjBuilder::new();
    if let Ok(proj_data) = std::env::var("PROJ_DATA") {
        builder.set_search_paths(PathBuf::from(proj_data)).unwrap();
    } else {
        let exe_root = std::env::current_exe().unwrap();
        exe_root.parent().unwrap();
        if !exe_root.clone().join("proj.db").exists() {
            match std::env::consts::OS {
                "windows" => {
                    let mut db_file =
                        std::fs::File::create("proj.db").unwrap();
                    db_file.write_all(PROJ_DB).unwrap();
                }
                _ => (),
            }
        }
        builder
            .set_search_paths(std::env::current_exe().unwrap().parent().unwrap())
            .unwrap();
    }
    builder
}
