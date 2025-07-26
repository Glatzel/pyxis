use std::sync::LazyLock;

use assert_cmd::Command;
use predicates::prelude::*;
const A: f64 = 4.0;
const OA: f64 = 1.0;
const OB: f64 = 2.0;
const DEGREES: f64 = 30.0;
static RADIANS: LazyLock<f64> = LazyLock::new(|| 30.0f64.to_radians());
static A1: LazyLock<f64> =
    LazyLock::new(|| (A - OA) * RADIANS.cos() - (A - OB) * RADIANS.sin() + OA);
static B1: LazyLock<f64> =
    LazyLock::new(|| (A - OA) * RADIANS.sin() + (A - OB) * RADIANS.cos() + OB);
#[test]
fn test_rotate_degrees_0() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args(["abacus", "-x", "2", "-y", "4", "-z", "6"])
        .args(["rotate", "--value", "0", "-p", "xy", "-u", "degrees"])
        .assert()
        .success()
      ;
}
#[test]
fn test_rotate_radians_0() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args(["abacus", "-x", "2", "-y", "4", "-z", "6"])
        .args(["rotate", "--value", "0", "-p", "xy", "-u", "radians"])
        .assert()
        .success()
;
}
#[test]
fn test_rotate_equals_origin() {
    for i in ["xy", "yz", "zx"] {
        Command::cargo_bin("pyxis")
            .unwrap()
            .args(["abacus", "-x", "2", "-y", "4", "-z", "6"])
            .args([
                "rotate", "--value", "150", "-p", i, "-u", "radians", "--ox", "2", "--oy", "4",
                "--oz", "6",
            ])
            .assert()
            .success()
     ;
    }
}
#[test]
fn test_rotate_xy() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
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
            &DEGREES.to_string(),
            "-p",
            "xy",
            "-u",
            "degrees",
            "--ox",
            &OA.to_string(),
            "--oy",
            &OB.to_string(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "x: {}, y: {}, z: {}",
            *A1, *B1, A
        )));
}
#[test]
fn test_rotate_yz() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
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
            &DEGREES.to_string(),
            "-p",
            "yz",
            "-u",
            "degrees",
            "--oy",
            &OA.to_string(),
            "--oz",
            &OB.to_string(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "x: {}, y: {}, z: {}",
            A, *A1, *B1,
        )));
}
#[test]
fn test_rotate_zx() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
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
            &DEGREES.to_string(),
            "-p",
            "zx",
            "-u",
            "degrees",
            "--oz",
            &OA.to_string(),
            "--ox",
            &OB.to_string(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "x: {}, y: {}, z: {}",
            *B1, A, *A1,
        )));
}
