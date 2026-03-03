use assert_cmd::Command;
use rstest::rstest;

#[rstest]
#[case("absolute", "origin")]
#[case("absolute", "relative")]
#[case("origin", "relative")]
fn test_migrate_2d(#[case] given: &str, #[case] another: &str) {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "transform",
            "-x",
            &1.0.to_string(),
            "-y",
            &format!("-y={}", (-2.0).to_string()),
        ])
        .args([
            "migrate2d",
            "-g",
            given,
            "-a",
            another,
            "--another-x",
            &(-10.0).to_string(),
            &format!("--another-y={}", &20.0.to_string()),
            "-r",
            &150.0.to_string(),
            "-u",
            "degrees",
        ])
        .assert()
        .success();
    insta::assert_snapshot!(
        format!("test_migrate_2d-{given}-{another}"),
        String::from_utf8_lossy(cmd.get_output().stdout.as_slice())
    );
}
