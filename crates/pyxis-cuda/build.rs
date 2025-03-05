use glob::glob;
fn main() {
    #[cfg(target_os = "windows")]
    let new_path_env_var = format!(
        "{};{}",
        std::env::var("PATH").unwrap(),
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64"
    );
    #[cfg(target_os = "linux")]
    let new_path_env_var = std::env::var("PATH").unwrap();
    for cu_file in glob("./src_cuda/*.cu").expect("Failed to read glob pattern") {
        let cu_file = cu_file.unwrap();
        let output = std::process::Command::new("nvcc")
            .args(["--ptx", cu_file.to_str().unwrap()])
            .args([
                "-o",
                &format!("src/{}.ptx", cu_file.file_stem().unwrap().to_str().unwrap()),
            ])
            .env("PATH", new_path_env_var.clone())
            .output()
            .expect("Failed to execute script");
        println!("Stdout:/n{}", String::from_utf8_lossy(&output.stdout));
        if !output.status.success() {
            panic!("Build failed.",);
        }
    }
}
