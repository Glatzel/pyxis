use std::path::PathBuf;

use assert_cmd::Command;
fn get_example_exe(name: &str) -> PathBuf {
    let root = PathBuf::from(std::env::var("CARGO_WORKSPACE_DIR").unwrap());
    let mut exe = root.clone();
    #[cfg(target_os = "linux")]
    exe.push(format!("target/llvm-cov-target/debug/examples/{name}"));
    #[cfg(target_os = "windows")]
    exe.push(format!("target/llvm-cov-target/debug/examples/{name}.exe"));
    if !exe.exists() {
        exe = root.clone();
        #[cfg(target_os = "linux")]
        exe.push(format!("target/llvm-cov-target/debug/examples/{name}"));
        #[cfg(target_os = "windows")]
        exe.push(format!("target/llvm-cov-target/debug/examples/{name}.exe"));
    }
    exe
}
#[test]
fn test_example_convert() {
    let exe = get_example_exe("convert");
    Command::new(exe).assert().success();
}
#[test]
fn test_example_custom_coordinate() {
    let exe = get_example_exe("custom_coordinate");
    Command::new(exe).assert().success();
}
