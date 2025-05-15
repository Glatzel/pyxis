use std::env::{self};
#[allow(unused_imports)]
use std::path::PathBuf;

fn main() {
    let workspace_root = env::var("CARGO_WORKSPACE_DIR").unwrap();
    // run pixi install
    std::process::Command::new("pixi")
        .arg("install")
        .current_dir(&workspace_root)
        .output()
        .expect("Failed to execute script");

    // pkg-config
    #[cfg(target_os = "windows")]
    {
        let path = env::var("PATH").unwrap().to_string();
        let pkg_exe_dir = dunce::canonicalize("../../.pixi/envs/default/Library/bin")
            .unwrap()
            .to_string_lossy()
            .to_string();
        unsafe {
            env::set_var("PATH", format!("{pkg_exe_dir};{path}"));
        }
    }
    let default_pkg_config_path = match env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "windows" => dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/x64-windows-static/lib/pkgconfig"
        ))
        .unwrap()
        .to_string_lossy()
        .to_string(),
        "linux" => dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/x64-linux-release/lib/pkgconfig"
        ))
        .unwrap()
        .to_string_lossy()
        .to_string(),
        "macos" => dunce::canonicalize(format!(
            "{workspace_root}/.pixi/envs/default/proj/arm64-osx-release/lib/pkgconfig"
        ))
        .unwrap()
        .to_string_lossy()
        .to_string(),
        other => {
            panic!("Unsupported OS: {}", other)
        }
    };
    if env::var("PKG_CONFIG_PATH").is_err() {
        unsafe {
            env::set_var("PKG_CONFIG_PATH", &default_pkg_config_path);
        }
    }

    // check LIBCLANG_PATH
    #[cfg(target_os = "windows")]
    match env::var("LIBCLANG_PATH") {
        Ok(path) => println!("Found `LIBCLANG_PATH`: {path}"),
        Err(_) => {
            let path = "C:/Program Files/LLVM/bin";
            if PathBuf::from(path).exists() {
                println!("cargo:rustc-env=LIBCLANG_PATH={path}");
            } else {
                panic!("`LIBCLANG_PATH` not found.");
            }
        }
    };

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
            println!("cargo:rustc-link-lib=static={}", lib);
            println!("Link to `{}`", lib);
            pklib
        }
        Err(e) => panic!("cargo:warning=Pkg-config error: {:?}", e),
    }
}
