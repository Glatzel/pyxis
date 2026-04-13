use assert_cmd::Command;
#[test]
fn test_translate() {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["transform"])
        .args(["translate", "--tx", "1", "--ty", "2", "--tz", "3"])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
