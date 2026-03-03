use assert_cmd::Command;
#[test]
fn test_proj() {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "transform",
            "-x",
            "4760096.421921",
            "-y",
            "3744293.729449",
            "-z",
            "0",
        ])
        .args(["proj", "--from", "EPSG:2230", "--to", "EPSG:26946"])
        .assert()
        .success();
    insta::with_settings!({filters => vec![
        (r"\d+\.\d+", "<coordinate>"),
    ]}, {
        insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
    });
}
