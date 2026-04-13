use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub idx: u8,
    pub method: String,
    pub parameter: serde_json::Value,

    pub output_x: f64,
    pub output_y: f64,
    pub output_z: f64,
    pub output_x_name: String,
    pub output_y_name: String,
    pub output_z_name: String,
}
