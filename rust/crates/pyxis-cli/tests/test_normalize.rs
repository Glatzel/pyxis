use assert_cmd::Command;
#[test]
fn test_normalize() {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["transform", "-x", "1", "-y", "2", "-z", "3"])
        .args(["normalize"])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
