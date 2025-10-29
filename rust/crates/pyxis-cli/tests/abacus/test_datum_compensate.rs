use assert_cmd::Command;
use predicates::prelude::*;
#[test]
fn test_datum_compensate() {
    Command::new(assert_cmd::cargo_bin!("pyxis"))

        .args(["abacus", "-x", "469704.6693", "-y", "2821940.796"])
        .args([
            "datum-compensate",
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
        .stdout(predicate::str::contains(
            "x: 469706.56912942487, y: 2821763.831232311",
        ));
}
