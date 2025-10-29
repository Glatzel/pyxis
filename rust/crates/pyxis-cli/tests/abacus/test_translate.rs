use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_translate() {
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["abacus"])
        .args(["translate", "--tx", "1", "--ty", "2", "--tz", "3"])
        .assert()
        .success()
        .stdout(predicate::str::contains("x: 1, y: 2, z: 3"));
}
