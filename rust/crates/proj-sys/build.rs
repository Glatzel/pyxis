use std::env::{self};
#[allow(unused_imports)]
use std::path::PathBuf;

use path_slash::PathBufExt;

fn main() {
    // 1. Explicit override wins
    let proj_root = env::var("PROJ_ROOT").map(PathBuf::from);

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
    let search_roots: Vec<PathBuf> = proj_root
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

    // Feature-gated: use generated (bindgen at build time) vs vendored (checked-in
    // src/bindings.rs)
    #[cfg(feature = "bindgen")]
    {
        generate_bindings(&proj_root);
        println!("cargo:rustc-cfg=feature=\"bindgen\"");
    }
}

#[cfg(feature = "bindgen")]
fn generate_bindings(proj_root: &PathBuf) {
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
    bindings
        .write_to_file("./src/bindings.rs")
        .expect(format!("Couldn't write bindings to './src/bindings.rs' !"));
    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect(format!(
            "Couldn't write bindings to {}",
            env::var("OUT_DIR").unwrap()
        ));
}
