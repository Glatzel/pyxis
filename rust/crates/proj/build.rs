#[allow(unused_imports)]
use std::path::PathBuf;

fn main() {
    println!("cargo:rustc-cfg=UPDATE=true");
    #[cfg(feature = "bindgen")]
    main_wrapper();
}
#[cfg(feature = "bindgen")]
fn main_wrapper() {
    // check LIBCLANG_PATH
    #[cfg(target_os = "windows")]
    match std::env::var("LIBCLANG_PATH") {
        Ok(path) => println!("Found `LIBCLANG_PATH`: {path}"),
        Err(_) => {
            let path = "C:/Program Files/LLVM/bin";

            if PathBuf::from(path).exists() {
                unsafe {
                    std::env::set_var("LIBCLANG_PATH", path);
                }
                println!("Set `LIBCLANG_PATH` to: {path}")
            } else {
                panic!("`LIBCLANG_PATH` not found.");
            }
        }
    };

    // Link
    let _pk_proj = link_lib("proj", "proj");

    // generate bindings
    if std::env::var("BINDGEN").unwrap_or("false".to_string()) == "true" {
        let header = &_pk_proj.include_paths[0]
            .join("proj.h")
            .to_string_lossy()
            .to_string();
        let bindings = bindgen::Builder::default()
            .header(header)
            .size_t_is_usize(true)
            .blocklist_type("max_align_t")
            .ctypes_prefix("libc")
            .use_core()
            .generate()
            .unwrap();
        if std::env::var("UPDATE").unwrap_or("false".to_string()) == "true" {
            bindings
                .write_to_file("./src/proj_sys/bindings.rs")
                .expect("Couldn't write bindings!");
        } else {
            println!("cargo:rustc-cfg=UPDATE=false");
            bindings
                .write_to_file(PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs"))
                .expect("Couldn't write bindings!");
        }
    }
}
#[cfg(feature = "bindgen")]
fn link_lib(name: &str, lib: &str) -> pkg_config::Library {
    match pkg_config::Config::new().probe(name) {
        Ok(pklib) => {
            println!("cargo:rustc-link-lib=static={}", lib);
            println!("Link to `{}`", lib);
            pklib
        }
        Err(e) => panic!("cargo:warning=Pkg-config error: {:?}", e),
    }
}
