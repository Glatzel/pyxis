use assert_cmd::Command;
use rstest::rstest;

#[rstest]
#[case("cartesian", "cylindrical")]
#[case("cartesian", "spherical")]
#[case("cylindrical", "cartesian")]
#[case("cylindrical", "spherical")]
#[case("spherical", "cylindrical")]
#[case("spherical", "cartesian")]
fn test_space(#[case] from: &str, #[case] to: &str) {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "transform",
            "-x",
            &1.2.to_string(),
            "-y",
            &3.4.to_string(),
            "-z",
            &5.6.to_string(),
        ])
        .args(["space", "-f", from, "-t", to])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
