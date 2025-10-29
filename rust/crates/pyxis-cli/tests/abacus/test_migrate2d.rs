use assert_cmd::Command;
use predicates::prelude::*;

const PARAM: (f64, f64, f64, f64, f64) = (10.0, 20.0, 2.0, -1.0, 150.0);

#[test]
fn test_rel_2d() {
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-x",
            &PARAM.2.to_string(),
            &format!("-y={}", &PARAM.3.to_string()),
        ])
        .args([
            "migrate2d",
            "-g",
            "absolute",
            "-a",
            "origin",
            "--another-x",
            &PARAM.0.to_string(),
            &format!("--another-y={}", &PARAM.1.to_string()),
            "-r",
            &PARAM.4.to_string(),
            "-u",
            "degrees",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains((-3.5717967697244886).to_string()))
        .stdout(predicate::str::contains(22.186533479473212.to_string()));
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-x",
            &PARAM.0.to_string(),
            &format!("-y={}", &PARAM.1.to_string()),
        ])
        .args([
            "migrate2d",
            "-g",
            "origin",
            "-a",
            "absolute",
            "--another-x",
            &PARAM.2.to_string(),
            &format!("--another-y={}", &PARAM.3.to_string()),
            "-r",
            &PARAM.4.to_string(),
            "-u",
            "degrees",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains((-3.5717967697244886).to_string()))
        .stdout(predicate::str::contains(22.186533479473212.to_string()));
}
#[test]
fn test_abs_2d() {
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-x",
            &PARAM.2.to_string(),
            &format!("-y={}", &PARAM.3.to_string()),
        ])
        .args([
            "migrate2d",
            "-g",
            "relative",
            "-a",
            "origin",
            "--another-x",
            &PARAM.0.to_string(),
            &format!("--another-y={}", &PARAM.1.to_string()),
            "-r",
            &PARAM.4.to_string(),
            "-u",
            "degrees",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(8.767949192431123.to_string()))
        .stdout(predicate::str::contains(21.866025403784437.to_string()));
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-x",
            &PARAM.0.to_string(),
            &format!("-y={}", &PARAM.1.to_string()),
        ])
        .args([
            "migrate2d",
            "-g",
            "origin",
            "-a",
            "relative",
            "--another-x",
            &PARAM.2.to_string(),
            &format!("--another-y={}", &PARAM.3.to_string()),
            "-r",
            &PARAM.4.to_string(),
            "-u",
            "degrees",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(8.767949192431123.to_string()))
        .stdout(predicate::str::contains(21.866025403784437.to_string()));
}
#[test]
fn test_origin_2d() {
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-x",
            &PARAM.2.to_string(),
            &format!("-y={}", &PARAM.3.to_string()),
        ])
        .args([
            "migrate2d",
            "-g",
            "relative",
            "-a",
            "absolute",
            "--another-x",
            &PARAM.0.to_string(),
            &format!("--another-y={}", &PARAM.1.to_string()),
            "-r",
            &PARAM.4.to_string(),
            "-u",
            "degrees",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(11.232050807568877.to_string()))
        .stdout(predicate::str::contains(18.133974596215563.to_string()));
    Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "abacus",
            "-x",
            &PARAM.0.to_string(),
            &format!("-y={}", &PARAM.1.to_string()),
        ])
        .args([
            "migrate2d",
            "-g",
            "absolute",
            "-a",
            "relative",
            "--another-x",
            &PARAM.2.to_string(),
            &format!("--another-y={}", &PARAM.3.to_string()),
            "-r",
            &PARAM.4.to_string(),
            "-u",
            "degrees",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(11.232050807568877.to_string()))
        .stdout(predicate::str::contains(18.133974596215563.to_string()));
}
