use assert_cmd::Command;
use predicates::prelude::*;
use pyxis::crypto::{BD09_LAT, BD09_LON, GCJ02_LAT, GCJ02_LON, WGS84_LAT, WGS84_LON};
#[test]
fn test_wgs84_to_gcj02() {
    Command::cargo_bin("pyxis-abacus")
        .unwrap()
        .args([
            "transform",
            "-x",
            &WGS84_LON.to_string(),
            "-y",
            &WGS84_LAT.to_string(),
        ])
        .args(["crypto", "-f", "wgs84", "-t", "gcj02"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            GCJ02_LON.to_string().get(0..17).unwrap(),
        ))
        .stdout(predicate::str::contains(
            GCJ02_LAT.to_string().get(0..17).unwrap(),
        ));
}
#[test]
fn test_gcj02_to_wgs84() {
    Command::cargo_bin("pyxis-abacus")
        .unwrap()
        .args([
            "transform",
            "-x",
            &GCJ02_LON.to_string(),
            "-y",
            &GCJ02_LAT.to_string(),
        ])
        .args(["crypto", "-f", "gcj02", "-t", "wgs84"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            WGS84_LON.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            WGS84_LAT.to_string().get(0..10).unwrap(),
        ));
}
#[test]
fn test_wgs84_to_bd09() {
    Command::cargo_bin("pyxis-abacus")
        .unwrap()
        .args([
            "transform",
            "-x",
            &WGS84_LON.to_string(),
            "-y",
            &WGS84_LAT.to_string(),
        ])
        .args(["crypto", "-f", "wgs84", "-t", "bd09"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            BD09_LON.to_string().get(0..17).unwrap(),
        ))
        .stdout(predicate::str::contains(
            BD09_LAT.to_string().get(0..17).unwrap(),
        ));
}
#[test]
fn test_bd09_to_wgs84() {
    Command::cargo_bin("pyxis-abacus")
        .unwrap()
        .args([
            "transform",
            "-x",
            &BD09_LON.to_string(),
            "-y",
            &BD09_LAT.to_string(),
        ])
        .args(["crypto", "-f", "bd09", "-t", "wgs84"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            WGS84_LON.to_string().get(0..8).unwrap(),
        ))
        .stdout(predicate::str::contains(
            WGS84_LAT.to_string().get(0..8).unwrap(),
        ));
}
#[test]
fn test_bd09_to_gcj02() {
    Command::cargo_bin("pyxis-abacus")
        .unwrap()
        .args([
            "transform",
            "-x",
            &BD09_LON.to_string(),
            "-y",
            &BD09_LAT.to_string(),
        ])
        .args(["crypto", "-f", "bd09", "-t", "gcj02"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            GCJ02_LON.to_string().get(0..17).unwrap(),
        ))
        .stdout(predicate::str::contains(
            GCJ02_LAT.to_string().get(0..17).unwrap(),
        ));
}
#[test]
fn test_gcj02_to_bd09() {
    Command::cargo_bin("pyxis-abacus")
        .unwrap()
        .args([
            "transform",
            "-x",
            &GCJ02_LON.to_string(),
            "-y",
            &GCJ02_LAT.to_string(),
        ])
        .args(["crypto", "-f", "gcj02", "-t", "bd09"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            BD09_LON.to_string().get(0..17).unwrap(),
        ))
        .stdout(predicate::str::contains(
            BD09_LAT.to_string().get(0..17).unwrap(),
        ));
}
