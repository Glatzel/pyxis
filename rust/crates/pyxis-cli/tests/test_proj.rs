use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_proj() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
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
        .stdout(predicate::str::contains("x: 1450880"))
        .stdout(predicate::str::contains("y: 1141263"));
}
