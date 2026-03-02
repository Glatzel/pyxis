use assert_cmd::Command;
use rstest::rstest;

#[test]
fn test_rotate_degrees_0() {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args(["rotate", "--value", "0", "-p", "xy", "-u", "degrees"])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
#[test]
fn test_rotate_radians_0() {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args(["rotate", "--value", "0", "-p", "xy", "-u", "radians"])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
#[rstest]
#[case("xy")]
#[case("yz")]
#[case("zx")]
fn test_rotate_equals_origin(#[case] axis: &str) {
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args(["transform", "-x", "2", "-y", "4", "-z", "6"])
        .args([
            "rotate", "--value", "150", "-p", axis, "-u", "radians", "--ox", "2", "--oy", "4",
            "--oz", "6",
        ])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
#[rstest]
#[case("xy")]
#[case("yz")]
#[case("zx")]
fn test_rotate(#[case] axis: &str) {
    let mut chars = axis.chars();
    let cmd = Command::new(assert_cmd::cargo_bin!("pyxis"))
        .args([
            "transform",
            "-x",
            &4.0.to_string(),
            "-y",
            &4.0.to_string(),
            "-z",
            &4.0.to_string(),
        ])
        .args([
            "rotate",
            "--value",
            &30.0.to_string(),
            "-p",
            axis,
            "-u",
            "degrees",
            &format!("--o{:?}", chars.next()),
            &1.0.to_string(),
            &format!("--o{:?}", chars.next()),
            &2.0.to_string(),
        ])
        .assert()
        .success();
    insta::assert_snapshot!(String::from_utf8_lossy(cmd.get_output().stdout.as_slice()));
}
