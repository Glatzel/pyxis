use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use pyxis_cli::Settings;
use schemars::schema_for;
fn main() {
    let schema = schema_for!(Settings);
    let json = serde_json::to_string_pretty(&schema).unwrap();
    let mut file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    file_path.push("res/pyxis-schema.json");
    let mut file = File::create(file_path).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}
