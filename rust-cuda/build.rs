use std::env;
use std::path::Path;

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
        let nvcc_exe_dir = dunce::canonicalize(".pixi/envs/default/Library/bin")
            .unwrap()
            .to_string_lossy()
            .to_string();

        let cl_paths =
            glob("C:/Program Files/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/bin/Hostx64/x64")
                .expect("Failed to read glob pattern")
                .filter_map(Result::ok)
                .collect::<Vec<std::path::PathBuf>>();

        let path = env::var("PATH").unwrap().to_string();
        unsafe {
            env::set_var(
                "PATH",
                format!("{nvcc_exe_dir};{};{path}", cl_paths.first().unwrap().to_string_lossy()),
            )
        };
        println!("{}", env::var("PATH").unwrap())

        let include_paths =
            glob("C:/Program Files/Microsoft Visual Studio/2022/*/VC/Tools/MSVC/*/include")
                .expect("Failed to read glob pattern")
                .filter_map(Result::ok)
                .collect::<Vec<std::path::PathBuf>>();
        unsafe { env::set_var("INCLUDE", format!("{:?}", include_paths.first().unwrap())) };
    }
    if cfg!(target_os = "linux") {
        let nvcc_exe_dir = dunce::canonicalize(".pixi/envs/default/bin")
            .unwrap()
            .to_string_lossy()
            .to_string();
        let path = env::var("PATH").unwrap().to_string();
        unsafe { env::set_var("PATH", format!("{nvcc_exe_dir}:{path}")) }
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
    println!("Stdout:n{}", String::from_utf8_lossy(&output.stdout));
    println!("Stderr:{}", String::from_utf8_lossy(&output.stderr));
    if !output.status.success() {
        panic!("Build failed.",);
    }
}
