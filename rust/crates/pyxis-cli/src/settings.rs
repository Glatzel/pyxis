use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use std::{fs, io};

use directories::ProjectDirs;
use miette::IntoDiagnostic;
use serde::{Deserialize, Serialize};

pub static SETTINGS: LazyLock<Mutex<Settings>> = LazyLock::new(|| Mutex::new(Settings::default()));
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Settings {
    pub trail_settings: crate::cli::trail::settings::Settings,
}

impl Settings {
    /// Initialise the config: try to read `term‑nmea.toml` located
    /// in the same directory as the executable.
    /// Falls back to `Config::default()` if the file is missing
    /// or malformed.
    pub fn overwrite_trail_settings(
        &mut self,
        port: Option<String>,
        baud_rate: Option<u32>,
        capacity: Option<usize>,
    ) -> miette::Result<()> {
        let path = Self::path();
        // Read from file or fallback
        let mut settings = match fs::read_to_string(&path) {
            Ok(content) => toml::from_str(&content).unwrap_or_else(|e| {
                clerk::warn!("Malformed config file: {e}. Using defaults.");
                Self::default()
            }),
            Err(e) => {
                clerk::warn!("Failed to read config: {e}. Using defaults.");
                Self::default()
            }
        };

        // Override with CLI args
        if let Some(port) = port {
            settings.trail_settings.port = port;
        }
        if let Some(baud) = baud_rate {
            settings.trail_settings.baud_rate = baud;
        }
        if let Some(cap) = capacity {
            settings.trail_settings.capacity = cap;
        }

        self.save()?;
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

    pub fn save(&self) -> miette::Result<()> {
        let toml_str = toml::to_string_pretty(self)
            .map_err(|e| io::Error::other(format!("TOML serialize error: {e}")))
            .into_diagnostic()?;

        let path = Self::path();

        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| io::Error::other(format!("Failed to create config directory: {e}")))
                .into_diagnostic()?;
        }

        fs::write(path, toml_str)
            .map_err(|e| io::Error::other(format!("Failed to write config: {e}")))
            .into_diagnostic()
    }
}
