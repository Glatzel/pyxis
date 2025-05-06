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
    let _pk_proj = link_lib("proj", "proj");

    // generate bindings
    #[cfg(feature = "update")]
    {
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
            .derive_debug(true)
            .generate()
            .unwrap();

        bindings
            .write_to_file(PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs"))
            .expect("Couldn't write bindings!");
        //only allow linux bindgen

        if std::env::var("UPDATE").unwrap_or("false".to_string()) == "true" {
            bindings
                .write_to_file("./src/bindings.rs")
                .expect("Couldn't write bindings!");
        }
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
