use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_translate() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args(["transform", "-x", "0", "-y", "0", "-z", "0"])
        .args(["translate", "-x", "1", "-y", "2", "-z", "3"])
        .assert()
        .success()
        .stdout(predicate::str::contains("x: 1, y: 2, z: 3"));
}
