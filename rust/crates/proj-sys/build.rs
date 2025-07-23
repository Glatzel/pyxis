use std::env::{self};
#[allow(unused_imports)]
use std::path::PathBuf;

fn main() {
    // Link
    let pk_proj = link_lib("proj", "proj");

    //bindgen
    if env::var("UPDATE").unwrap_or("false".to_string()) != "true"
        && env::var("BINDGEN").unwrap_or("false".to_string()) != "true"
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

    if env::var("UPDATE").unwrap_or("false".to_string()) == "true" {
        bindings
            .write_to_file("./src/bindings.rs")
            .expect("Couldn't write bindings!");
    }
    if env::var("BINDGEN").unwrap_or("false".to_string()) == "true" {
        println!("cargo:rustc-cfg=bindgen");
        bindings
            .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}
fn link_lib(name: &str, lib: &str) -> pkg_config::Library {
    match pkg_config::Config::new().probe(name) {
        Ok(pklib) => {
            println!("cargo:rustc-link-lib=static={lib}");
            println!("Link to `{lib}`");
            pklib
        }
        Err(e) => panic!("cargo:warning=Pkg-config error: {e:?}"),
    }
}
