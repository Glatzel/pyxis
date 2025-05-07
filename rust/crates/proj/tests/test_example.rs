use std::path::PathBuf;

use assert_cmd::Command;
#[test]
fn test_example_convert() {
    let root = PathBuf::from(std::env::var("CARGO_WORKSPACE_DIR").unwrap());
    let mut exe = root.clone();
    exe.push("target/llvm-cov-target/debug/examples/convert.exe");
    if !exe.exists() {
        exe = root.clone();
        exe.push("target/debug/examples/convert.exe")
    }
    Command::new(exe).current_dir(root).assert().success();
}
