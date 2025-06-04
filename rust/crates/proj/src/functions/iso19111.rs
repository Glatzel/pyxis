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
mod proj_advanced;
mod proj_basic;

///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_string_list_destroy>
fn string_list_destroy(ptr: *mut *mut i8) {
    unsafe {
        proj_sys::proj_string_list_destroy(ptr);
    }
}
///# See Also
///
/// * [`crate::Proj::identify`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_int_list_destroy>
fn _proj_int_list_destroy() { unimplemented!("Use other function to instead.") }
///# See Also
///
/// * [`crate::Context::get_celestial_body_list_from_database`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_celestial_body_list_from_database>
fn _celestial_body_list_destroy() { unimplemented!("Use other function to instead.") }
///# See Also
///
/// * [`crate::Context::get_crs_info_list_from_database`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_crs_list_parameters_create>
fn _get_crs_list_parameters_create() { unimplemented!("Use other function to instead.") }
///# See Also
///
/// * [`crate::Context::get_crs_info_list_from_database`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_crs_list_parameters_destroy>
fn _get_crs_list_parameters_destroy() { unimplemented!("Use other function to instead.") }
///# See Also
///
/// * [`crate::Context::get_crs_info_list_from_database`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_info_list_destroy>
fn _crs_info_list_destroy() { unimplemented!("Use other function to instead.") }
///# See Also
///
/// * [`crate::Context::get_units_from_database`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_unit_list_destroy>
fn _unit_list_destroy() { unimplemented!("Use other function to instead.") }

///# See Also
///
/// * [`crate::extension::pj_obj_list_to_vec`]
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_get_count>
fn _proj_list_get_count() { unimplemented!("Use other function to instead.") }
///# See Also
///
/// * [`crate::extension::pj_obj_list_to_vec`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_destroy>
fn _proj_list_destroy() { unimplemented!("Use other function to instead.") }
