use std::fs;
use std::path::PathBuf;
use std::sync::LazyLock;

use directories::ProjectDirs;
use parking_lot::Mutex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::cli;
const DEFAULT_SETTINGS_STR: &str = include_str!("../res/pyxis-default.toml");

pub static SETTINGS: LazyLock<Mutex<Settings>> = LazyLock::new(|| {
    let path = Settings::path();
    // Read from file or fallback
    let settings = match fs::read_to_string(&path) {
        Ok(content) => toml::from_str(&content).unwrap_or_else(|e| {
            clerk::warn!("Malformed config file: {e}. Using defaults.");
            Settings::default()
        }),
        Err(e) => {
            clerk::warn!("Failed to read config: {e}. Using defaults.");
            Settings::default()
        }
    };
    Mutex::new(settings)
});
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Settings {
    pub trail: crate::cli::trail::settings::Settings,
}

impl Settings {
    pub fn overwrite_settings(args: &cli::SubCommands) -> mischief::Result<()> {
        let mut settings = SETTINGS.lock();
        match *args {
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
}
impl Default for Settings {
    fn default() -> Self {
        toml::from_str(DEFAULT_SETTINGS_STR).expect("Failed to parse embedded default settings")
    }
}
