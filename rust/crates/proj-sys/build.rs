use std::env;
use std::path::PathBuf;

fn main() {
    // Read env vars
    let lib_dir = env::var("PROJ_LIB_DIR").expect("PROJ_LIB_DIR not set");
    let include_dir = env::var("PROJ_INCLUDE_DIR").expect("PROJ_INCLUDE_DIR not set");

    // Link the library statically (or dynamically if preferred)
    println!("cargo:rustc-link-search=native={lib_dir}");
    println!("cargo:rustc-link-lib=static=proj"); // use `dylib=proj` for dynamic
    println!("cargo:rustc-link-lib=m"); // proj often depends on `libm`

    // Tell cargo to rerun if these env vars change
    println!("cargo:rerun-if-env-changed=PROJ_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PROJ_INCLUDE_DIR");

    // bindgen logic
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
