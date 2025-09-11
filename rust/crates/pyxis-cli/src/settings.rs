use std::fs;
use std::path::PathBuf;
use std::sync::LazyLock;

use directories::ProjectDirs;
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::cli;
const DEFAULT_SETTINGS_STR: &str = include_str!("../res/pyxis-default.toml");
const SETTINGS_SCHEMA_STR: &str = include_str!("../res/pyxis-schema.json");
pub static SETTINGS: LazyLock<Mutex<Settings>> = LazyLock::new(|| {
    let path = Settings::path();
    // Read from file or fallback
    let settings = match fs::read_to_string(&path) {
        Ok(content) => {
            Settings::validate_against_schema(&content).unwrap();
            toml::from_str(&content).unwrap_or_else(|e| {
                clerk::warn!("Malformed config file: {e}. Using defaults.");
                Settings::default()
            })
        }
        Err(e) => {
            clerk::warn!("Failed to read config: {e}. Using defaults.");
            Settings::default()
        }
    };
    Mutex::new(settings)
});
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct Settings {
    pub abacus: crate::cli::abacus::Settings,
    pub trail: crate::cli::trail::settings::Settings,
}

impl Settings {
    pub fn overwrite_settings(args: &cli::SubCommands) -> mischief::Result<()> {
        let mut settings = SETTINGS.lock();
        match *args {
            cli::SubCommands::Abacus { output_format, .. } => {
                output_format.inspect(|o| settings.abacus.output_format = *o);
            }
            cli::SubCommands::Trail {
                ref port,
                baud_rate,
                capacity,
            } => {
                port.clone().inspect(|p| settings.trail.port = p.clone());
                baud_rate.inspect(|b| settings.trail.baud_rate = *b);
                capacity.inspect(|c| settings.trail.capacity = *c);
            }
        }
        Ok(())
    }

    pub fn path() -> PathBuf {
        if let Some(proj_dirs) = ProjectDirs::from("", "", "pyxis") {
            proj_dirs.config_dir().join("pyxis.toml")
        } else {
            clerk::warn!("Cannot determine config directory. Using local file.");
            PathBuf::from("pyxis.toml")
        }
    }

    pub fn save(&self) -> mischief::Result<()> {
        let toml_str = toml::to_string_pretty(self)?;

        let path = Self::path();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        Ok(fs::write(path, toml_str)?)
    }
    fn validate_against_schema(toml_str: &str) -> mischief::Result<()> {
        // 1. Parse TOML to a generic JSON `Value`
        let json: Value = toml::from_str::<Value>(toml_str)?;

        // 2. Parse the JSON schema
        let schema: Value = serde_json::from_str(SETTINGS_SCHEMA_STR)?;
        let schema_static: &'static Value = Box::leak(Box::new(schema));

        // 3. Compile the schema
        let compiled = jsonschema::draft202012::new(schema_static)?;

        // 4. Validate the data
        let result = compiled.validate(&json);
        if let Err(errors) = result {
            clerk::error!("Validation error: {}", errors);
            clerk::error!("Validation failed");
            mischief::bail!("Validation failed");
        } else {
            Ok(())
        }
    }
}
impl Default for Settings {
    fn default() -> Self {
        toml::from_str(DEFAULT_SETTINGS_STR).expect("Failed to parse embedded default settings")
    }
}
#[cfg(test)]
mod test {
    use super::*;
    const INVALID_TOML: &str = r#"
port = "COM3"
baud_rate = 115200
capacity = 500
"#;
    #[test]
    fn test_valid_toml_parsing() -> mischief::Result<()> {
        Settings::validate_against_schema(DEFAULT_SETTINGS_STR)?;
        Ok(())
    }
    #[test]
    fn test_invalid_toml_parsing() -> mischief::Result<()> {
        let result = Settings::validate_against_schema(INVALID_TOML);
        assert!(result.is_err());
        Ok(())
    }
}
