use serde::{Deserialize, Serialize};

use crate::cli::abacus::OutputFormat;
#[derive(Debug, Serialize, Deserialize)]
pub struct Settings {
    pub output_format: OutputFormat,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::Plain,
        }
    }
}
