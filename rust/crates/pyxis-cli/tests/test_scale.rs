use assert_cmd::Command;
#[test]
fn test_scale() {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args([
            "scale", "--sx", "2", "--sy", "2", "--sz", "2", "--ox", "1", "--oy", "2", "--oz", "3",
        ])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
