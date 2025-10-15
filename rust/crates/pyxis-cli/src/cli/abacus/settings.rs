use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::cli::abacus::OutputFormat;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub struct Settings {
    pub output_format: OutputFormat,
}
