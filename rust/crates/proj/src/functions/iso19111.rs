//!The PJ* objects returned by proj_create_from_wkt(),
//! proj_create_from_database() and other functions in that section will have
//! generally minimal interaction with the functions declared in the previous
//! sections (calling those functions on those objects will either return an
//! error or default/nonsensical values). The exception is for ISO19111 objects
//! of type CoordinateOperation that can be exported as a valid PROJ pipeline.
//! In this case, objects will work for example with proj_trans_generic().
//! Conversely, objects returned by proj_create() and proj_create_argv(), which
//! are not of type CRS (can be tested with proj_is_crs()), will return an error
//! when used with functions of this section.
//!
//! # References
//!
//! * <https://proj.org/en/stable/development/reference/functions.html#transformation-setup>

mod context_advanced;
mod context_basic;
mod insert_object_session;
mod operation_factory_context;
mod proj_advanced;
mod proj_basic;
mod proj_obj_list;

/// use [`crate::data_types::iso19111::CrsListParameters::new`] instead
fn _get_crs_list_parameters_create() { unimplemented!() }

/// rust automatically drops the
/// [`crate::data_types::iso19111::CrsListParameters`] when it goes out of scope
fn _get_crs_list_parameters_destroy() { unimplemented!() }

/// directly called in related functions
fn _int_list_destroy() { unimplemented!() }

/// directly called in related functions
fn _list_destroy() { unimplemented!() }

/// directly called in related functions
fn _string_destroy() { unimplemented!() }

/// directly called in related functions
fn _string_list_destroy() { unimplemented!() }

/// directly called in related functions
fn _unit_list_destroy() { unimplemented!() }
