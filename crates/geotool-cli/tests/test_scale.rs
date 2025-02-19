use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_scale() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args([
            "scale", "--sx", "2", "--sy", "2", "--sz", "2", "--ox", "1", "--oy", "2", "--oz", "3",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("x: 3, y: 6, z: 9"));
}
