use std::path::PathBuf;
use std::{env, fs};

fn main() {
    let lib_dir = PathBuf::from(env::var("LIB_DIR").expect("LIB_DIR not set"));
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rerun-if-env-changed=LIB_DIR");

    // Detect all static libraries in the directory
    for entry in fs::read_dir(&lib_dir).expect("Cannot read LIB_DIR") {
        let entry = entry.expect("Invalid entry");
        let path = entry.path();

        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            match ext {
                "a" => {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        println!(
                            "cargo:rustc-link-lib=static={}",
                            name.trim_start_matches("lib")
                        );
                    }
                }
                "lib" => {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        println!("cargo:rustc-link-lib=static={}", name);
                    }
                }
                _ => {}
            }
        }
    }

    // Bindgen section (if needed)
    let include_dir = PathBuf::from(env::var("INCLUDE_DIR").expect("INCLUDE_DIR not set"));
    let do_update = env::var("UPDATE").unwrap_or_default() == "true";
    let do_bindgen = env::var("BINDGEN").unwrap_or_default() == "true";

    if !do_update && !do_bindgen {
        return;
    }

    let header = PathBuf::from(&include_dir).join("proj.h");

    let bindings = bindgen::Builder::default()
        .header(header.to_string_lossy())
        .size_t_is_usize(true)
        .blocklist_type("max_align_t")
        .ctypes_prefix("libc")
        .use_core()
        .generate()
        .expect("Failed to generate bindings");

    if do_update {
        bindings
            .write_to_file("./src/bindings.rs")
            .expect("Couldn't write bindings!");
    }

    if do_bindgen {
        println!("cargo:rustc-cfg=bindgen");
        bindings
            .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}
