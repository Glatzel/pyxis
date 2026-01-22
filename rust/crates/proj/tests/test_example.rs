use std::path::PathBuf;

use assert_cmd::Command;
fn get_example_exe(name: &str) -> mischief::Result<PathBuf> {
    let root = PathBuf::from(std::env::var("CARGO_WORKSPACE_DIR")?);
    let mut exe = root.clone();
    #[cfg(not(target_os = "windows"))]
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
    Ok(exe)
}
#[test]
fn test_example_convert() -> mischief::Result<()> {
    let exe = get_example_exe("convert")?;
    Command::new(exe).assert().success();
    Ok(())
}
#[test]

fn test_example_custom_coordinate() -> mischief::Result<()> {
    let exe = get_example_exe("custom_coordinate")?;
    Command::new(exe).assert().success();
    Ok(())
}
