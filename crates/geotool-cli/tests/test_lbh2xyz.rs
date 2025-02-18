use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_lbh2xyz() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args(["transform", "-x", "48.8566", "-y", "2.3522", "-z", "35.0"])
        .args(["lbh2xyz", "-a", "6378137", "--invf", "298.257223563"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "x: 4192979.6198897623, y: 4799159.563725418, z: 260022.66015989496",
        ));
}
