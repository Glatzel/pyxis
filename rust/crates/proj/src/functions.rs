mod area_of_interest;
#[cfg(feature = "unrecommended")]
mod cleanup;
mod coordinate_transformation;
mod custom_io;
pub mod distances;
mod error_reporting;
pub mod info_functions;
///The PJ* objects returned by proj_create_from_wkt(),
/// proj_create_from_database() and other functions in that section will have
/// generally minimal interaction with the functions declared in the previous
/// sections (calling those functions on those objects will either return an
/// error or default/nonsensical values). The exception is for ISO19111 objects
/// of type CoordinateOperation that can be exported as a valid PROJ pipeline.
/// In this case, objects will work for example with proj_trans_generic().
/// Conversely, objects returned by proj_create() and proj_create_argv(), which
/// are not of type CRS (can be tested with proj_is_crs()), will return an error
/// when used with functions of this section.
///
/// # References
///
///<https://proj.org/en/stable/development/reference/functions.html#transformation-setup>
mod iso19111;
pub mod lists;
mod logging;
mod network;
mod threading_contexts;
///The objects returned by the functions defined in this section have minimal
/// interaction with the functions of the C API for ISO-19111 functionality, and
/// vice versa. See its introduction paragraph for more details.
///
/// # References
///
///<https://proj.org/en/stable/development/reference/functions.html#c-api-for-iso-19111-functionality>
mod transformation_setup;
pub mod various;
// pub use iso19111::*;
#[cfg(feature = "unrecommended")]
pub use cleanup::*;
