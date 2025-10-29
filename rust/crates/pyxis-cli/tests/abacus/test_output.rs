use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_output_plain_no_name() {
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-x",
            "4760096.421921",
            "-y",
            "3744293.729449",
            "-z",
            "0",
            "-o",
            "verbose",
        ])
        .args(["proj", "--from", "EPSG:2230", "--to", "EPSG:26946"])
        .assert()
        .success()
        .stdout(predicate::str::contains("EPSG:2230"))
        .stdout(predicate::str::contains("EPSG:26946"));
}
#[test]
fn test_output_plain_with_name() {
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-n",
            "Test",
            "-x",
            "4760096.421921",
            "-y",
            "3744293.729449",
            "-z",
            "0",
            "-o",
            "verbose",
        ])
        .args(["proj", "--from", "EPSG:2230", "--to", "EPSG:26946"])
        .assert()
        .success()
        .stdout(predicate::str::contains("EPSG:2230"))
        .stdout(predicate::str::contains("EPSG:26946"));
}
#[test]
fn test_json() {
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-n",
            "Test",
            "-x",
            "4760096.421921",
            "-y",
            "3744293.729449",
            "-z",
            "0",
            "-o",
            "json",
        ])
        .args(["proj", "--from", "EPSG:2230", "--to", "EPSG:26946"])
        .assert()
        .success()
        .stdout(predicate::str::contains("EPSG:2230"))
        .stdout(predicate::str::contains("EPSG:26946"));
}
