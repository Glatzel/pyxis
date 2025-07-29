use serde::{Deserialize, Serialize};

mod tab_coord;
pub use tab_coord::TabCoordSettings;
#[derive(Debug, Serialize, Deserialize)]
pub struct Settings {
    pub port: String,
    pub baud_rate: u32,
    pub capacity: usize,

    pub tab_coord: TabCoordSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            port: "COM1".into(), // pick sensible defaults for your platform
            baud_rate: 9_600,
            capacity: 1000,

            tab_coord: TabCoordSettings {
                custom_cs: String::default(),
            },
        }
    }
}
