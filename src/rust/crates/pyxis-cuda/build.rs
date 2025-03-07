use dunce::canonicalize;
use glob::glob;
use path_slash::{self, PathExt};
use std::path::Path;
fn main() {
    let cpp_src_dir = canonicalize(Path::new("."))
        .unwrap()
        .parent()
        .unwrap() //crates
        .parent()
        .unwrap() //rust
        .parent()
        .unwrap() //src
        .join("cpp")
        .join("src");
    let cpp_include_dir = canonicalize(Path::new("."))
        .unwrap()
        .parent()
        .unwrap() //crates
        .parent()
        .unwrap() //rust
        .parent()
        .unwrap() //src
        .join("cpp")
        .join("include");
    let cu_kernel_dir = canonicalize(Path::new("."))
        .unwrap()
        .parent()
        .unwrap() //crates
        .parent()
        .unwrap() //rust
        .parent()
        .unwrap() //src
        .join("cuda")
        .join("kernels");
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
    let cu_files = glob(cu_kernel_dir.join("*.cu").to_str().unwrap())
        .expect("Failed to read glob pattern")
        .into_iter()
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
        .env("PATH", new_path_env_var.clone())
        .output()
        .expect("Failed to execute script");
    println!("Stdout:/n{}", String::from_utf8_lossy(&output.stdout));
    println!("Stderr:/n{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        panic!("Build failed.",);
    }
}
