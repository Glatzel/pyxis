use std::path::PathBuf;

use assert_cmd::Command;

pub fn get_example_exe(name: &str) -> PathBuf {
    let root = PathBuf::from(env!("CARGO_WORKSPACE_DIR"));
    let mut exe = root.clone();
    if cfg!(windows) {
        exe.push(format!("target/llvm-cov-target/debug/examples/{name}.exe"));
    } else {
        exe.push(format!("target/llvm-cov-target/debug/examples/{name}"));
    }
    exe
}
#[test]
fn test_bayer_image() {
    let exe = get_example_exe("jiaxing_bump_station");
    Command::new(exe).assert().success();
}
#[test]
fn test_zhengyong_expressway_dehua_east_interchange() {
    let exe = get_example_exe("zhengyong_expressway_dehua_east_interchange");
    Command::new(exe).assert().success();
}
