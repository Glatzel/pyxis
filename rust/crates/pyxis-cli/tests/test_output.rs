use assert_cmd::Command;
use rstest::rstest;
#[rstest]
#[case("simple")]
#[case("verbose")]
#[case("json")]
fn test_output(#[case] output: &str) {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "transform",
            "-x",
            "4760096.421921",
            "-y",
            "3744293.729449",
            "-z",
            "0",
            "-o",
            output,
        ])
        .args(["proj", "--from", "EPSG:2230", "--to", "EPSG:26946"])
        .assert()
        .success();
    insta::assert_snapshot!(
        format!("test_output-{output}"),
        String::from_utf8_lossy(cmd.get_output().stdout.as_slice())
    );
}
