use std::env;
use std::path::Path;

use dunce::canonicalize;
use glob::glob;
use path_slash::PathExt;
fn main() {
    // run pixi install
    std::process::Command::new("pixi")
        .arg("install")
        .current_dir(env::var("CARGO_WORKSPACE_DIR").unwrap())
        .output()
        .expect("Failed to execute script");
    // env
    if cfg!(target_os = "windows") {
        println!(
            "cargo:rustc-env=INCLUDE=C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/include"
        );
    }
    if env::var("CUDA_PATH").is_err() {
        let cuda_path = dunce::canonicalize("../../.pixi/envs/default/Library")
            .unwrap()
            .to_string_lossy()
            .to_string();
        println!("cargo:rustc-env=CUDA_PATH={cuda_path}");
        let path = env::var("PATH").unwrap().to_string();
        let nvcc_exe_dir = dunce::canonicalize("../../.pixi/envs/default/Library/bin")
            .unwrap()
            .to_string_lossy()
            .to_string();
        let cuda_library_path = dunce::canonicalize("../../.pixi/envs/default/Library/lib")
            .unwrap()
            .to_string_lossy()
            .to_string();
        if cfg!(target_os = "windows") {
            let cl_path = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64";
            let include = "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/include";
            println!("cargo:rustc-env=PATH={nvcc_exe_dir};{cl_path};{path}");
            println!("cargo:rustc-env=INCLUDE={include}");
        } else {
            println!("cargo:rustc-env=CUDA_LIBRARY_PATH={cuda_library_path}");
        }
    }
    //set src code dir
    let cpp_src_dir = canonicalize(Path::new("."))
        .unwrap()
        .parent()
        .unwrap() //root
        .join("cpp")
        .join("src");
    let cpp_include_dir = canonicalize(Path::new("."))
        .unwrap()
        .parent()
        .unwrap() //root
        .join("cpp")
        .join("include");
    let cu_kernel_dir = canonicalize(Path::new("."))
        .unwrap()
        .parent()
        .unwrap() //root
        .join("cuda")
        .join("src");
    println!("cargo:rerun-if-changed={}", cpp_src_dir.to_str().unwrap());
    println!("cargo:rerun-if-changed={}", cu_kernel_dir.to_str().unwrap());
    let cu_files = glob(cu_kernel_dir.join("*.cu").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .map(|f| {
            canonicalize(f.unwrap())
                .unwrap()
                .to_slash_lossy()
                .to_string()
        })
        .collect::<Vec<String>>();
    let output = std::process::Command::new("nvcc")
        .arg("-fmad=false")
        .args(["-I", cpp_src_dir.to_slash_lossy().to_string().as_str()])
        .args(["-I", cpp_include_dir.to_slash_lossy().to_string().as_str()])
        .arg("--ptx")
        .args(cu_files)
        .args(["-odir", "./src"])
        .output()
        .expect("Failed to execute script");
    println!("Stdout:/n{}", String::from_utf8_lossy(&output.stdout));
    println!("Stderr:/n{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        panic!("Build failed.",);
    }
}
