use dunce::canonicalize;
use glob::glob;
use path_slash::{self, PathExt};
use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};
fn main() {
    let cpp_src_dir = canonicalize(
        Path::new(".")
            .parent()
            .unwrap() //pyxis-cuda
            .parent()
            .unwrap() //src
            .join("cpp")
            .join("src"),
    )
    .unwrap();
    let cu_kernel_dir = canonicalize(
        Path::new(".")
            .parent()
            .unwrap() //pyxis-cuda
            .parent()
            .unwrap() //src
            .join("cuda")
            .join("kernels")
            .canonicalize()
            .unwrap(),
    )
    .unwrap();
    println!("cargo:rerun-if-changed={}", cpp_src_dir.to_str().unwrap());
    println!("cargo:rerun-if-changed={}", cu_kernel_dir.to_str().unwrap());
    #[cfg(target_os = "windows")]
    let new_path_env_var = format!(
        "{};{};{}",
        std::env::var("PATH").unwrap(),
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64",
        "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64"
    );
    #[cfg(target_os = "linux")]
    let new_path_env_var = std::env::var("PATH").unwrap();
    let cu_files =
        glob(cu_kernel_dir.join("*.cu").to_str().unwrap()).expect("Failed to read glob pattern");
    let cpp_files =
        glob(cpp_src_dir.join("*.cpp").to_str().unwrap()).expect("Failed to read glob pattern");
    let files = cu_files
        .into_iter()
        .chain(cpp_files)
        .map(|f| {
            canonicalize(f.unwrap())
                .unwrap()
                .to_slash_lossy()
                .to_string()
        })
        .collect::<Vec<String>>();

    let output = std::process::Command::new("nvcc")
        .arg("-fmad=false")
        .arg("--ptx")
        .args(files)
        .args(["-odir", "./src"])
        .env("PATH", new_path_env_var.clone())
        .output()
        .expect("Failed to execute script");
    println!("Stdout:/n{}", String::from_utf8_lossy(&output.stdout));
    println!("Stderr:/n{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        panic!("Build failed.",);
    }
}
