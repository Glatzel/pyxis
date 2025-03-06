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
    let cu_files = glob("./src_cuda/*.cu").expect("Failed to read glob pattern");
    let h_files = glob("./src_cuda/*.h").expect("Failed to read glob pattern");
    let files = cu_files
        .into_iter()
        .chain(h_files)
        .map(|f| f.unwrap().to_string_lossy().into_owned())
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
    if !output.status.success() {
        panic!("Build failed.",);
    }
}
