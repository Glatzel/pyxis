use std::env;

fn main() {
    // PROJ_DATA
    let default_proj_data = match env::var("CARGO_CFG_TARGET_OS").unwrap().as_str() {
        "windows" => {
            dunce::canonicalize("../../.pixi/envs/default/proj/x64-windows-static/share/proj")
                .unwrap()
                .to_string_lossy()
                .to_string()
        }
        "linux" => {
            dunce::canonicalize("../../.pixi/envs/default/proj/x64-linux-release/share/proj")
                .unwrap()
                .to_string_lossy()
                .to_string()
        }
        "macos" => {
            dunce::canonicalize("../../.pixi/envs/default/proj/arm64-osx-release/share/proj")
                .unwrap()
                .to_string_lossy()
                .to_string()
        }
        other => {
            panic!("Unsupported OS: {other}")
        }
    };
    if env::var("PROJ_DATA").is_err() {
        unsafe {
            env::set_var("PROJ_DATA", &default_proj_data);
        }
        println!("cargo:rustc-env=PROJ_DATA={default_proj_data}");
    }
}
