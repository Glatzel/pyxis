use assert_cmd::Command;
use predicates::prelude::*;

const CARTESIAN: (f64, f64, f64) = (1.2, 3.4, -5.6);
const CYLINDRICAL: (f64, f64, f64) = (3.605551275463989, 1.2315037123408519, -5.60000000000000);
const SPHERICAL: (f64, f64, f64) = (1.2315037123408519, 2.5695540653144073, 6.660330322138685);

#[test]
fn test_cartesian_to_cylindrical() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
            "-x",
            &CARTESIAN.0.to_string(),
            "-y",
            &CARTESIAN.1.to_string(),
            "-z",
            &CARTESIAN.2.to_string(),
        ])
        .args(["space", "-f", "cartesian", "-t", "cylindrical"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            CYLINDRICAL.0.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            CYLINDRICAL.1.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(CYLINDRICAL.2.to_string()));
}
#[test]
fn test_cartesian_to_spherical() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
            "-x",
            &CARTESIAN.0.to_string(),
            "-y",
            &CARTESIAN.1.to_string(),
            "-z",
            &CARTESIAN.2.to_string(),
        ])
        .args(["space", "-f", "cartesian", "-t", "spherical"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            SPHERICAL.0.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            SPHERICAL.1.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            SPHERICAL.2.to_string().get(0..10).unwrap(),
        ));
}
#[test]
fn test_cylindrical_to_cartesian() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
            "-x",
            &CYLINDRICAL.0.to_string(),
            "-y",
            &CYLINDRICAL.1.to_string(),
            "-z",
            &CYLINDRICAL.2.to_string(),
        ])
        .args(["space", "-f", "cylindrical", "-t", "cartesian"])
        .assert()
        .success()
        .stdout(predicate::str::contains(CARTESIAN.0.to_string()))
        .stdout(predicate::str::contains(CARTESIAN.1.to_string()))
        .stdout(predicate::str::contains(CARTESIAN.2.to_string()));
}
#[test]
fn test_cylindrical_to_spherical() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
            "-x",
            &CYLINDRICAL.0.to_string(),
            "-y",
            &CYLINDRICAL.1.to_string(),
            "-z",
            &CYLINDRICAL.2.to_string(),
        ])
        .args(["space", "-f", "cylindrical", "-t", "spherical"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            SPHERICAL.0.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            SPHERICAL.1.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            SPHERICAL.2.to_string().get(0..10).unwrap(),
        ));
}
#[test]
fn test_spherical_to_cartesian() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
            "-x",
            &SPHERICAL.0.to_string(),
            "-y",
            &SPHERICAL.1.to_string(),
            "-z",
            &SPHERICAL.2.to_string(),
        ])
        .args(["space", "-f", "spherical", "-t", "cartesian"])
        .assert()
        .success()
        .stdout(predicate::str::contains(CARTESIAN.0.to_string()))
        .stdout(predicate::str::contains(CARTESIAN.1.to_string()))
        .stdout(predicate::str::contains("-5.5999".to_string()));
}
#[test]
fn test_spherical_to_cylindrical() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "abacus",
            "-x",
            &SPHERICAL.0.to_string(),
            "-y",
            &SPHERICAL.1.to_string(),
            "-z",
            &SPHERICAL.2.to_string(),
        ])
        .args(["space", "-f", "spherical", "-t", "cylindrical"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            CYLINDRICAL.0.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            CYLINDRICAL.1.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains("-5.5999999".to_string()));
}
