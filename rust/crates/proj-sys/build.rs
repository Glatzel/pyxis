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

    // === Link all static libraries in LIB_DIR ===
    println!("cargo:rustc-link-search=native={lib_dir}");
    // Explicitly link the static libs in correct order
    println!("cargo:rustc-link-lib=static=proj");
    println!("cargo:rustc-link-lib=static=sqlite3");
    println!("cargo:rustc-link-lib=static=curl");
    println!("cargo:rustc-link-lib=static=tiff");

    // Link `libm` on Unix-like platforms
    let target = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target == "linux" {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=stdc++");
    }
    if target == "macos" {
        println!("cargo:rustc-link-lib=m");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=c++");
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
