use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_xyz2lbh() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "transform",
            "-x",
            "4192979.6198897623",
            "-y",
            "4799159.563725418",
            "-z",
            "260022.66015989496",
        ])
        .args(["xyz2lbh", "-a", "6378137", "--invf", "298.257223563"])
        .assert()
        .success()
        .stdout(predicate::str::contains("longitude: 48.8566"))
        .stdout(predicate::str::contains("latitude: 2.35219"))
        .stdout(predicate::str::contains("elevation: 35"));
}
