use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str::FromStr;

use envoy::{CStrListToVecString, CStrToString, ToCStr};
use miette::IntoDiagnostic;

use super::string_list_destroy;
use crate::data_types::iso19111::*;
use crate::{Context, OPTION_NO, OPTION_YES, Proj, ProjOptions, check_result, pj_obj_list_to_vec};
impl Context {
    ///# References
    ///
    /// <>
    fn create_operation_factory_context(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_desired_accuracy(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_area_of_interest(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_area_of_interest_name(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_crs_extent_use(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_spatial_criterion(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_grid_availability_use(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_use_proj_alternative_grid_names(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_allow_use_intermediate_crs(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_allowed_intermediate_crs(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_discard_superseded(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_allow_ballpark_transformations(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _create_operations(&self) { todo!() }
}
impl OperationFactoryContext {}
