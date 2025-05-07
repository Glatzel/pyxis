use assert_cmd::Command;
use predicates::prelude::*;
#[cfg(not(target_os = "macos"))]
#[test]
fn test_output_plain_no_name() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "transform",
            "-x",
            "4760096.421921",
            "-y",
            "3744293.729449",
            "-z",
            "0",
            "-o",
            "plain",
        ])
        .args(["proj", "--from", "EPSG:2230", "--to", "EPSG:26946"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            r#"
x: 4760096.421921, y: 3744293.729449, z: 0
|-- step: 1
|-- method: proj
|-- parameter:
|       {
|         "from": "EPSG:2230",
|         "to": "EPSG:26946"
|       }
▼
x: 1450880.2910605022, y: 1141263.0111604782, z: 0"#,
        ));
}
#[cfg(not(target_os = "macos"))]
#[test]
fn test_output_plain_with_name() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "transform",
            "-n",
            "Test",
            "-x",
            "4760096.421921",
            "-y",
            "3744293.729449",
            "-z",
            "0",
            "-o",
            "plain",
        ])
        .args(["proj", "--from", "EPSG:2230", "--to", "EPSG:26946"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            r#"
x: 4760096.421921, y: 3744293.729449, z: 0
|-- step: 1
|-- method: proj
|-- parameter:
|       {
|         "from": "EPSG:2230",
|         "to": "EPSG:26946"
|       }
▼
x: 1450880.2910605022, y: 1141263.0111604782, z: 0"#,
        ));
}
#[cfg(not(target_os = "macos"))]
#[test]
fn test_json() {
    Command::cargo_bin("pyxis")
        .unwrap()
        .args([
            "transform",
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
        .stdout(predicate::str::contains(
            r#"{
  "name": "Test",
  "record": [
    {
      "idx": 0,
      "method": "input",
      "output_x": 4760096.421921,
      "output_x_name": "x",
      "output_y": 3744293.729449,
      "output_y_name": "y",
      "output_z": 0.0,
      "output_z_name": "z",
      "parameter": {}
    },
    {
      "idx": 1,
      "method": "proj",
      "output_x": 1450880.2910605022,
      "output_x_name": "x",
      "output_y": 1141263.0111604782,
      "output_y_name": "y",
      "output_z": 0.0,
      "output_z_name": "z",
      "parameter": {
        "from": "EPSG:2230",
        "to": "EPSG:26946"
      }
    }
  ]
}"#,
        ));
}
