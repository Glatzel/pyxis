use assert_cmd::Command;
use pyxis::crypto::{BD09_LAT, BD09_LON, GCJ02_LAT, GCJ02_LON, WGS84_LAT, WGS84_LON};
use rstest::rstest;
#[rstest]
#[case("wgs84", "gcj02", WGS84_LON, WGS84_LAT)]
#[case("gcj02", "wgs84", GCJ02_LON, GCJ02_LAT)]
#[case("gcj02", "bd09", GCJ02_LON, GCJ02_LAT)]
#[case("wgs84", "bd09", WGS84_LON, WGS84_LAT)]
#[case("bd09", "gcj02", BD09_LON, BD09_LAT)]
#[case("bd09", "wgs84", BD09_LON, BD09_LAT)]
fn test_crypto(#[case] from: &str, #[case] to: &str, #[case] lon: f64, #[case] lat: f64) {
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
