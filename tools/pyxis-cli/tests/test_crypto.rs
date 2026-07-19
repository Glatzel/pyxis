use assert_cmd::Command;
use rstest::rstest;
#[rstest]
#[case("wgs84", "gcj02")]
#[case("gcj02", "wgs84")]
#[case("gcj02", "bd09")]
#[case("wgs84", "bd09")]
#[case("bd09", "gcj02")]
#[case("bd09", "wgs84")]
fn test_crypto(#[case] from: &str, #[case] to: &str) {
    let lon = 120.0;
    let lat = 30.0;
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["transform", "-x", &lon.to_string(), "-y", &lat.to_string()])
        .args(["crypto", "-f", from, "-t", to])
        .assert()
        .success();
    insta::assert_snapshot!(
        format!("test_crypto-{from}-{to}"),
        String::from_utf8_lossy(cmd.get_output().stdout.as_slice())
    );
}
