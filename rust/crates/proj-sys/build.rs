use std::env;
#[allow(unused_imports)]
use std::path::PathBuf;

use path_slash::PathBufExt;

fn main() {
    // 1. Explicit override wins
    let proj_root = env::var("PROJ_ROOT").map(PathBuf::from).unwrap_or_default();

    // 2. CMAKE_PREFIX_PATH entries (colon-separated on Unix, semicolon on Windows)
    let cmake_prefixes: Vec<PathBuf> = env::var("CMAKE_PREFIX_PATH")
        .unwrap_or_default()
        .split(if cfg!(windows) { ';' } else { ':' })
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();

    // 3. System fallback roots
    let system_roots: Vec<PathBuf> = vec![PathBuf::from("/usr/local"), PathBuf::from("/usr")];

    // Build ordered search list: PROJ_ROOT first, then CMAKE_PREFIX_PATH, then
    // system
    let search_roots: Vec<PathBuf> = vec![proj_root.clone()]
        .into_iter()
        .chain(cmake_prefixes)
        .chain(system_roots)
        .collect();

    for root in &search_roots {
        let lib_dir = root.join("lib");
        if lib_dir.exists() {
            println!(
                "cargo:rustc-link-search=native={}",
                lib_dir.to_slash_lossy()
            );
        }
    }

    println!("cargo:rustc-link-lib=proj");

    if env::var("UPDATE_PROJ_BINDINGS").is_ok() {
        let include_dir = proj_root.join("include");
        let header = include_dir.join("proj.h").to_slash_lossy().to_string();

        let bindings = bindgen::Builder::default()
            .header(header)
            .size_t_is_usize(true)
            .blocklist_type("max_align_t")
            .ctypes_prefix("libc")
            .use_core()
            .generate()
            .unwrap();
        match env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
            "windows" => {
                bindings
                    .write_to_file("./src/bindings-win.rs")
                    .expect("Couldn't write bindings!");
            }
            "linux" => match env::var("CARGO_CFG_TARGET_ARCH").unwrap().as_str() {
                "x86_64" => {
                    bindings
                        .write_to_file("./src/bindings-linux.rs")
                        .expect("Couldn't write bindings!");
                }
                "aarch64" => {
                    bindings
                        .write_to_file("./src/bindings-linux-aarch64.rs")
                        .expect("Couldn't write bindings!");
                }
                other => {
                    panic!("Unsupported OS: {other}")
                }
            },
            "macos" => {
                bindings
                    .write_to_file("./src/bindings-macos.rs")
                    .expect("Couldn't write bindings!");
            }
            other => {
                panic!("Unsupported OS: {other}")
            }
        }
    }
}
