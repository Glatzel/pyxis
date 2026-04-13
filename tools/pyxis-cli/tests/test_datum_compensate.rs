use assert_cmd::Command;
#[test]
fn test_datum_compensate() {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["transform", "-x", "469704.6693", "-y", "2821940.796"])
        .args([
            "datum-compensate",
            "--hb",
            "400",
            "--radius",
            "6378137",
            "--x0",
            "500000",
            "--y0",
            "0",
        ])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
