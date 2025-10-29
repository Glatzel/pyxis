use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_normalize() {
    let length: f64 = (1.0f64.powi(2) + 2.0f64.powi(2) + 3.0f64.powi(2)).sqrt();
    Command::new(assert_cmd::cargo_bin!("pyxis")
        .unwrap()
        .args(["abacus", "-x", "1", "-y", "2", "-z", "3"])
        .args(["normalize"])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "x: {}, y: {}, z: {}",
            1.0 / length,
            2.0 / length,
            3.0 / length,
        )));
}
