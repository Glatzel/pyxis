#[allow(unused_imports)]
use std::path::PathBuf;

use tracing::level_filters::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
fn main() {
    tracing_subscriber::registry()
        .with(clerk::terminal_layer(LevelFilter::DEBUG, true))
        .init();

    // check LIBCLANG_PATH
    #[cfg(target_os = "windows")]
    match std::env::var("LIBCLANG_PATH") {
        Ok(path) => tracing::info!("Found `LIBCLANG_PATH`: {path}"),
        Err(_) => {
            let path = "C:/Program Files/LLVM/bin";

            if PathBuf::from(path).exists() {
                unsafe {
                    std::env::set_var("LIBCLANG_PATH", path);
                }
                tracing::info!("Set `LIBCLANG_PATH` to: {path}")
            } else {
                tracing::error!("`LIBCLANG_PATH` not found.");
                panic!("`LIBCLANG_PATH` not found.");
            }
        }
    };

    // Link
    let _pk_proj = link_lib("proj", "proj");

    // generate bindings
    #[cfg(any(feature = "bindgen", feature = "update"))]
    {
        let header = &_pk_proj.include_paths[0]
            .join("proj.h")
            .to_string_lossy()
            .to_string();
        let bindings = bindgen::Builder::default()
            .header(header)
            .use_core()
            .size_t_is_usize(true)
            .blocklist_type("max_align_t")
            // .ctypes_prefix("libc")
            .generate()
            .unwrap();

        bindings
            .write_to_file(PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs"))
            .expect("Couldn't write bindings!");
        #[cfg(feature = "update")]
        bindings
            .write_to_file("./src/bindings.rs")
            .expect("Couldn't write bindings!");
        eprintln!(
            "Build bingings to: {:?}",
            PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.rs")
        );
    }
}
fn link_lib(name: &str, lib: &str) -> pkg_config::Library {
    match pkg_config::Config::new().probe(name) {
        Ok(pklib) => {
            println!("cargo:rustc-link-lib=static={}", lib);
            clerk::info!("Link to `{}`", lib);
            pklib
        }
        Err(e) => panic!("cargo:warning=Pkg-config error: {:?}", e),
    }
}
