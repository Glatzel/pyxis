use assert_cmd::Command;
use predicates::prelude::*;
use std::sync::LazyLock;
const A: f64 = 4.0;
const OA: f64 = 1.0;
const OB: f64 = 2.0;
const ANGLE: f64 = 30.0;
static RADIANS: LazyLock<f64> = LazyLock::new(|| 30.0f64.to_radians());
static A1: LazyLock<f64> =
    LazyLock::new(|| (A - OA) * RADIANS.cos() - (A - OB) * RADIANS.sin() + OA);
static B1: LazyLock<f64> =
    LazyLock::new(|| (A - OA) * RADIANS.sin() + (A - OB) * RADIANS.cos() + OB);
#[test]
fn test_rotate_angle_0() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args(["rotate", "--value", "0", "-p", "xy", "-u", "angle"])
        .assert()
        .success()
        .stderr(predicate::str::contains("WARN"));
}
#[test]
fn test_rotate_radians_0() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args(["rotate", "--value", "0", "-p", "xy", "-u", "radians"])
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
                "rotate", "--value", "150", "-p", i, "-u", "radians", "--ox", "2", "--oy", "4",
                "--oz", "6",
            ])
            .assert()
            .success()
            .stderr(predicate::str::contains("WARN"));
    }
}
#[test]
fn test_rotate_xy() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args([
            "transform",
            "-x",
            &A.to_string(),
            "-y",
            &A.to_string(),
            "-z",
            &A.to_string(),
        ])
        .args([
            "rotate",
            "--value",
            &ANGLE.to_string(),
            "-p",
            "xy",
            "-u",
            "angle",
            "--ox",
            &OA.to_string(),
            "--oy",
            &OB.to_string(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "x: {}, y: {}, z: {}",
            A1.to_string(),
            B1.to_string(),
            A
        )));
}
#[test]
fn test_rotate_yz() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args([
            "transform",
            "-x",
            &A.to_string(),
            "-y",
            &A.to_string(),
            "-z",
            &A.to_string(),
        ])
        .args([
            "rotate",
            "--value",
            &ANGLE.to_string(),
            "-p",
            "yz",
            "-u",
            "angle",
            "--oy",
            &OA.to_string(),
            "--oz",
            &OB.to_string(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "x: {}, y: {}, z: {}",
            A,
            A1.to_string(),
            B1.to_string(),
        )));
}
#[test]
fn test_rotate_zx() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args([
            "transform",
            "-x",
            &A.to_string(),
            "-y",
            &A.to_string(),
            "-z",
            &A.to_string(),
        ])
        .args([
            "rotate",
            "--value",
            &ANGLE.to_string(),
            "-p",
            "zx",
            "-u",
            "angle",
            "--oz",
            &OA.to_string(),
            "--ox",
            &OB.to_string(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "x: {}, y: {}, z: {}",
            B1.to_string(),
            A,
            A1.to_string(),
        )));
}
