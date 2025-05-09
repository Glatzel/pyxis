#![allow(clippy::too_many_arguments)]
/// This section describes the numerous data types in use in PROJ.
///
/// # References
///<https://proj.org/en/stable/development/reference/datatypes.html>
pub mod data_types;
pub mod extension;
/// Proj functions
///
/// # References
///<https://proj.org/en/stable/development/reference/functions.html>
pub mod functions;
/// Proj version
pub mod version;

pub use data_types::{Pj, PjArea, PjContext, PjDirection, PjLogLevel};
pub(crate) use extension::*;
pub use extension::{IPjCoord, PjParams};
