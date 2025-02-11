use assert_cmd::Command;
use predicates::prelude::*; // For more readable assertions

#[test]
fn test_main_with_args() {
    Command::cargo_bin("geotool")
        .unwrap()
        .arg("--help")
        .assert()
        .success();
}
