use std::env;
use std::path::{Path, PathBuf};

use dunce::canonicalize;
use glob::glob;
use path_slash::PathExt;
fn main() {
    // run pixi install
    std::process::Command::new("pixi")
        .arg("install")
        .output()
        .expect("Failed to execute script");
    // env
    if cfg!(target_os = "windows") {
        build_win()
    }
    if cfg!(target_os = "linux") {
        build_linux()
    }
}
fn get_src() -> (PathBuf, PathBuf, Vec<String>) {
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
    (cpp_src_dir, cpp_include_dir, cu_files)
}
fn build_win() {
    //set path
    let nvcc_exe_dir = dunce::canonicalize(".pixi/envs/default/Library/bin")
        .unwrap()
        .to_string_lossy()
        .to_string();
    let output = std::process::Command::new("pixi")
        .args([
            "run",
            "vswhere",
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            "**\\Hostx64\\x64",
        ])
        .output()
        .expect("Failed to execute script");
    let cl_path = String::from_utf8_lossy(&output.stdout).replace("\n", "");
    let path = env::var("PATH").unwrap().to_string();
    //set include
    let output = std::process::Command::new("pixi")
        .args([
            "run",
            "vswhere",
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-find",
            "VC\\Tools\\MSVC\\*\\include",
        ])
        .output()
        .expect("Failed to execute script");
    let include_path = String::from_utf8_lossy(&output.stdout);
    unsafe { env::set_var("INCLUDE", format!("{include_path}")) };
    //build
    let (cpp_src_dir, cpp_include_dir, cu_files) = get_src();
    let output = std::process::Command::new("nvcc")
        .arg("-fmad=false")
        .args(["-I", cpp_src_dir.to_slash_lossy().to_string().as_str()])
        .args(["-I", cpp_include_dir.to_slash_lossy().to_string().as_str()])
        .arg("--ptx")
        .args(cu_files)
        .args(["-odir", "./src"])
        .env("PATH", format!("{nvcc_exe_dir};{cl_path};{path}"))
        .output()
        .expect("Failed to execute script");
    println!("Stdout:n{}", String::from_utf8_lossy(&output.stdout));
    println!("Stderr:{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        panic!("Build failed.",);
    }
}
fn build_linux() {
    //set env
    let nvcc_exe_dir = dunce::canonicalize(".pixi/envs/default/bin")
        .unwrap()
        .to_string_lossy()
        .to_string();
    let path = env::var("PATH").unwrap().to_string();
    unsafe { env::set_var("PATH", format!("{nvcc_exe_dir}:{path}")) }
    //build
    let (cpp_src_dir, cpp_include_dir, cu_files) = get_src();
    let output = std::process::Command::new("nvcc")
        .arg("-fmad=false")
        .args(["-I", cpp_src_dir.to_slash_lossy().to_string().as_str()])
        .args(["-I", cpp_include_dir.to_slash_lossy().to_string().as_str()])
        .arg("--ptx")
        .args(cu_files)
        .args(["-odir", "./src"])
        .output()
        .expect("Failed to execute script");
    println!("Stdout:n{}", String::from_utf8_lossy(&output.stdout));
    println!("Stderr:{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        panic!("Build failed.",);
    }
}
