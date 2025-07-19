use std::path::PathBuf;
use std::{env, fs};

fn main() {
    // === Read environment variables ===
    let lib_dir = env::var("LIB_DIR").expect("LIB_DIR not set");
    let include_dir = env::var("INCLUDE_DIR").expect("INCLUDE_DIR not set");
    let do_update = env::var("UPDATE").unwrap_or_default() == "true";
    let do_bindgen = env::var("BINDGEN").unwrap_or_default() == "true";

    // === Instruct Cargo to rerun if env vars change ===
    println!("cargo:rerun-if-env-changed=LIB_DIR");
    println!("cargo:rerun-if-env-changed=INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=UPDATE");
    println!("cargo:rerun-if-env-changed=BINDGEN");

    // Link `libm` on Unix-like platforms
    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=stdc++");
    }
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=dl");
    }

    // === Link all static libraries in LIB_DIR ===
    println!("cargo:rustc-link-search=native={lib_dir}");
    for entry in fs::read_dir(&lib_dir).expect("Cannot read LIB_DIR") {
        let entry = entry.expect("Invalid entry");
        let path = entry.path();

        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            match ext {
                "a" => {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        // Remove "lib" prefix for Unix .a files
                        println!(
                            "cargo:rustc-link-lib=static={}",
                            name.trim_start_matches("lib")
                        );
                    }
                }
                "lib" => {
                    if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                        // MSVC static libraries
                        println!("cargo:rustc-link-lib=static={name}");
                    }
                }
                _ => {}
            }
        }
    }

    // === Skip bindgen unless explicitly requested ===
    if !do_update && !do_bindgen {
        return;
    }

    // === Generate bindings with bindgen ===
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
        let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}
