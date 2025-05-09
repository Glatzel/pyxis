#[allow(unused_imports)]
use std::path::PathBuf;

fn main() {
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
    let pk_proj = link_lib("proj", "proj");

    //bindgen
    if std::env::var("UPDATE").unwrap_or("false".to_string()) != "true"
        && std::env::var("BINDGEN").unwrap_or("false".to_string()) != "true"
    {
        return;
    }
    // generate bindings
    let header = &pk_proj.include_paths[0]
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
    }
    if std::env::var("BINDGEN").unwrap_or("false".to_string()) == "true" {
        println!("cargo:rustc-cfg=bindgen");
        bindings
            .write_to_file(PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}
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
