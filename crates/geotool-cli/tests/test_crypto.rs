use assert_cmd::Command;
use predicates::prelude::*; 

const WGS84_LON: f64 = 121.0917077;
const WGS84_LAT: f64 = 30.6107779;
const GCJ02_LON: f64 = 121.09626935575027;
const GCJ02_LAT: f64 = 30.608604331756705;
const BD09_LON: f64 = 121.10271732371203;
const BD09_LAT: f64 = 30.61484572185035;
#[test]
fn test_wgs84_to_gcj02() {
    Command::cargo_bin("geotool")
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
            GCJ02_LON.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            GCJ02_LAT.to_string().get(0..10).unwrap(),
        ));
}
#[test]
fn test_gcj02_to_wgs84_() {
    Command::cargo_bin("geotool")
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
    Command::cargo_bin("geotool")
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
            BD09_LON.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            BD09_LAT.to_string().get(0..10).unwrap(),
        ));
}
#[test]
fn test_bd09_to_wgs84() {
    Command::cargo_bin("geotool")
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
            WGS84_LON.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            WGS84_LAT.to_string().get(0..10).unwrap(),
        ));
}
#[test]
fn test_bd09_to_gcj02() {
    Command::cargo_bin("geotool")
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
            GCJ02_LON.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            GCJ02_LAT.to_string().get(0..10).unwrap(),
        ));
}
#[test]
fn test_gcj02_to_bd09() {
    Command::cargo_bin("geotool")
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
            BD09_LON.to_string().get(0..10).unwrap(),
        ))
        .stdout(predicate::str::contains(
            BD09_LAT.to_string().get(0..10).unwrap(),
        ));
}
