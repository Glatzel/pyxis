use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};
use std::{fs, io};

use directories::ProjectDirs;
use miette::IntoDiagnostic;
use serde::{Deserialize, Serialize};

use crate::cli;

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
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct Settings {
    pub abacus_settings: crate::cli::abacus::Settings,
    pub trail_settings: crate::cli::trail::settings::Settings,
}

impl Settings {
    pub fn overwrite_settings(args: &cli::SubCommands) -> miette::Result<()> {
        let mut settings: std::sync::MutexGuard<'_, Settings> = SETTINGS.lock().unwrap();
        match *args {
            cli::SubCommands::Abacus { output_format, .. } => {
                output_format.inspect(|o| settings.abacus_settings.output_format = *o);
            }
            cli::SubCommands::Trail {
                ref port,
                baud_rate,
                capacity,
            } => {
                port.clone()
                    .inspect(|p| settings.trail_settings.port = p.clone());
                baud_rate.inspect(|b| settings.trail_settings.baud_rate = *b);
                capacity.inspect(|c| settings.trail_settings.capacity = *c);
            }
        }

        settings.save()?;
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
