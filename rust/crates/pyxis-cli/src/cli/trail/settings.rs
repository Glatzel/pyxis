use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod tab_coord;
pub use tab_coord::TabCoordSettings;
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Settings {
    pub port: String,
    pub baud_rate: u32,
    pub capacity: usize,

    pub tab_coord: TabCoordSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            port: default_port(),
            baud_rate: 9_600,
            capacity: 1000,

            tab_coord: TabCoordSettings {
                custom_cs: String::default(),
            },
        }
    }
}
fn default_port() -> String {
    #[cfg(target_os = "windows")]
    {
        "COM1".to_string()
    }

    #[cfg(target_os = "linux")]
    {
        "/dev/ttyUSB0".to_string()
    }

    #[cfg(target_os = "macos")]
    {
        "/dev/cu.usbserial-0001".to_string()
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        "UNKNOWN_PORT".to_string()
    }
}
