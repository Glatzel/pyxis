use assert_cmd::Command;
use predicates::prelude::*;
use std::sync::LazyLock;
const A: f64 = 4.0;
const B: f64 = 4.0;
const OA: f64 = 1.0;
const OB: f64 = 2.0;
const ANGLE: f64 = 30.0;
static RADIANS: LazyLock<f64> = LazyLock::new(|| 30.0f64.to_radians());
#[test]
fn test_rotate_angle_0() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args(["rotate", "--value", "0", "-a", "xy", "-u", "angle"])
        .assert()
        .success()
        .stderr(predicate::str::contains("WARN"));
}
#[test]
fn test_rotate_radians_0() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args(["rotate", "--value", "0", "-a", "xy", "-u", "radians"])
        .assert()
        .success()
        .stderr(predicate::str::contains("WARN"));
}
#[test]
fn test_rotate_equals_origin() {
    for i in ["xy", "yz", "zx"] {
        Command::cargo_bin("geotool")
            .unwrap()
            .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
            .args([
                "rotate", "--value", "150", "-a", i, "-u", "radians", "--ox", "2", "--oy", "4",
                "--oz", "6",
            ])
            .assert()
            .success()
            .stderr(predicate::str::contains("WARN"));
    }
}
