use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_proj() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "-v",
            "-v",
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
        .success()
        .stdout(predicate::str::contains(
            "x: 1450880.2910605022, y: 1141263.0111604782, z: 0",
        ));
}
