fn main() {
    #[cfg(target_os = "windows")]
    let new_path = format!(
        "{};{}",
        std::env::var("PATH").unwrap(),
        "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64"
    );
    #[cfg(target_os = "linux")]
    let new_path = std::env::var("PATH").unwrap();
    let output = std::process::Command::new("nvcc")
        .args(["--ptx", "src_cuda/datum_compense.cu"])
        .args(["-o", "src/datum_compense.ptx"])
        .env("PATH", new_path)
        .output()
        .expect("Failed to execute script");
    println!("Stdout:/n{}", String::from_utf8_lossy(&output.stdout));
    if !output.status.success() {
        panic!("Build failed.",);
    }
}
