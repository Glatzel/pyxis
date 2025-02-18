use assert_cmd::Command;
use predicates::prelude::*; 
#[test]
fn test_datum_compense() {
    Command::cargo_bin("geotool")
        .unwrap()
        .args(["transform", "-x", "469704.6693", "-y", "2821940.796"])
        .args([
            "datum-compense",
            "--hb",
            "400",
            "--radius",
            "6378137",
            "--x0",
            "500000",
            "--y0",
            "0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("x: 469706.56912942487, y: 2821763.831232311"));
}
