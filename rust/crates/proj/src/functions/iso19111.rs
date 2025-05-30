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

use core::f64;
use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr::{self, null};
use std::result;

use miette::IntoDiagnostic;

use crate::data_types::iso19111::*;
use crate::{
    Context, OPTION_NO, OPTION_YES, Proj, ProjOptions, check_result, cstr_to_string,
    vec_cstr_to_string,
};

/// # ISO-19111 Base functions
impl crate::Context {
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_autoclose_database>
    #[deprecated]
    fn _set_autoclose_database(&self) { unimplemented!() }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_database_path>
    pub fn set_database_path(
        &self,
        db_path: &Path,
        aux_db_paths: Option<&[PathBuf]>,
    ) -> miette::Result<&Self> {
        let db_path = CString::new(db_path.to_string_lossy().to_string()).into_diagnostic()?;

        let aux_db_paths: Option<Vec<CString>> = aux_db_paths.map(|aux_db_paths| {
            aux_db_paths
                .iter()
                .map(|f| {
                    CString::new(f.to_string_lossy().to_string()).expect("Error creating CString")
                })
                .collect()
        });

        let aux_db_paths_ptr: Option<Vec<*const i8>> =
            aux_db_paths.map(|aux_db_paths| aux_db_paths.iter().map(|f| f.as_ptr()).collect());

        let result = unsafe {
            proj_sys::proj_context_set_database_path(
                self.ptr,
                db_path.as_ptr(),
                if let Some(aux_db_paths_ptr) = aux_db_paths_ptr {
                    aux_db_paths_ptr.as_ptr()
                } else {
                    ptr::null()
                },
                ptr::null(),
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(self)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_database_path>
    pub fn get_database_path(&self) -> PathBuf {
        PathBuf::from(
            cstr_to_string(unsafe { proj_sys::proj_context_get_database_path(self.ptr) })
                .unwrap_or_default(),
        )
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_database_metadata>
    pub fn get_database_metadata(&self, key: DatabaseMetadataKey) -> Option<String> {
        let key = CString::from(key);
        cstr_to_string(unsafe {
            proj_sys::proj_context_get_database_metadata(self.ptr, key.as_ptr())
        })
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_database_structure>
    pub fn get_database_structure(&self) -> miette::Result<Vec<String>> {
        let ptr = unsafe { proj_sys::proj_context_get_database_structure(self.ptr, ptr::null()) };
        let out_vec = vec_cstr_to_string(ptr).unwrap();
        string_list_destroy(ptr);
        Ok(out_vec)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_guess_wkt_dialect>
    pub fn guess_wkt_dialect(&self, wkt: &str) -> miette::Result<GuessedWktDialect> {
        GuessedWktDialect::try_from(unsafe {
            proj_sys::proj_context_guess_wkt_dialect(
                self.ptr,
                CString::new(wkt).expect("Error creating CString").as_ptr(),
            )
        })
        .into_diagnostic()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_from_wkt>
    pub fn create_from_wkt(
        &self,
        wkt: &str,
        strict: Option<bool>,
        unset_identifiers_if_incompatible_def: Option<bool>,
    ) -> miette::Result<Proj> {
        let mut options = ProjOptions::new(2);
        options.push_optional_pass(strict, "STRICT");
        options.push_optional_pass(
            unset_identifiers_if_incompatible_def,
            "UNSET_IDENTIFIERS_IF_INCOMPATIBLE_DEF",
        );
        let vec_ptr = options.vec_ptr();
        let mut out_warnings: *mut *mut i8 = std::ptr::null_mut();
        let mut out_grammar_errors: *mut *mut i8 = std::ptr::null_mut();
        let ptr = unsafe {
            proj_sys::proj_create_from_wkt(
                self.ptr,
                CString::new(wkt).into_diagnostic()?.as_ptr(),
                vec_ptr.as_ptr(),
                &mut out_warnings,
                &mut out_grammar_errors,
            )
        };
        //warning
        if let Some(warnings) = vec_cstr_to_string(out_warnings) {
            for w in warnings.iter() {
                clerk::warn!("{w}");
            }
        };
        //error
        if let Some(errors) = vec_cstr_to_string(out_grammar_errors) {
            for e in errors.iter() {
                clerk::warn!("{e}");
            }
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_from_database>
    pub fn create_from_database(
        &self,
        auth_name: &str,
        code: &str,
        category: Category,
        use_projalternative_grid_names: bool,
    ) -> miette::Result<Proj> {
        let auth_name = CString::new(auth_name).expect("Error creating CString");
        let code = CString::new(code).expect("Error creating CString");
        let ptr = unsafe {
            proj_sys::proj_create_from_database(
                self.ptr,
                auth_name.as_ptr(),
                code.as_ptr(),
                category.into(),
                use_projalternative_grid_names as i32,
                null(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_uom_get_info_from_database>
    pub fn uom_get_info_from_database(
        &self,
        auth_name: &str,
        code: &str,
    ) -> miette::Result<UomInfo> {
        let mut name: *const std::ffi::c_char = std::ptr::null();
        let mut conv_factor: f64 = f64::NAN;
        let mut category: *const std::ffi::c_char = std::ptr::null();
        let result = unsafe {
            proj_sys::proj_uom_get_info_from_database(
                self.ptr,
                CString::new(auth_name)
                    .expect("Error creating CString")
                    .as_ptr(),
                CString::new(code).expect("Error creating CString").as_ptr(),
                &mut name,
                &mut conv_factor,
                &mut category,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(UomInfo::new(
            cstr_to_string(name).unwrap(),
            conv_factor,
            UomCategory::try_from(unsafe { CString::from_raw(category.cast_mut()) })?,
        ))
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_get_info_from_database>
    pub fn _grid_get_info_from_database(&self, grid_name: &str) -> miette::Result<GridInfoDB> {
        let mut full_name: *const std::ffi::c_char = std::ptr::null();
        let mut package_name: *const std::ffi::c_char = std::ptr::null();
        let mut url: *const std::ffi::c_char = std::ptr::null();
        let mut direct_download: i32 = i32::default();
        let mut open_license: i32 = i32::default();
        let mut available: i32 = i32::default();
        let result = unsafe {
            proj_sys::proj_grid_get_info_from_database(
                self.ptr,
                CString::new(grid_name)
                    .expect("Error creating CString")
                    .as_ptr(),
                &mut full_name,
                &mut package_name,
                &mut url,
                &mut direct_download,
                &mut open_license,
                &mut available,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(GridInfoDB::new(
            cstr_to_string(full_name).unwrap(),
            cstr_to_string(package_name).unwrap(),
            cstr_to_string(url).unwrap(),
            direct_download != 0,
            open_license != 0,
            available != 0,
        ))
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_from_name>
    fn _create_from_name(&self) { unimplemented!() }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_non_deprecated>
    fn _get_non_deprecated(&self) { unimplemented!() }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_geoid_models_from_database>
    fn _get_geoid_models_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_authorities_from_database>
    fn _get_authorities_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_crs_info_list_from_database>
    fn _get_codes_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_celestial_body_list_from_database>
    fn _get_celestial_body_list_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_get_crs_info_list_from_database>
    fn _get_crs_info_list_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_units_from_database>
    fn _get_units_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _insert_object_session_create(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _insert_object_session_destroy(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_insert_statements(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _suggests_code_for(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_operation_factory_context(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_desired_accuracy(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_area_of_interest(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_area_of_interest_name(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_crs_extent_use(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_spatial_criterion(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_grid_availability_use(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_use_proj_alternative_grid_names(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_allow_use_intermediate_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_allowed_intermediate_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_discard_superseded(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _operation_factory_context_set_allow_ballpark_transformations(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_operations(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _list_get(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_suggested_operation(&self) { unimplemented!() }
}
/// # ISO-19111 Advanced functions
impl Context {
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_cs>
    pub fn create_cs(
        &self,
        coordinate_system_type: CoordinateSystemType,
        axis: &[AxisDescription],
    ) -> miette::Result<crate::Proj> {
        let axis_count = axis.len();
        let mut axis_vec: Vec<proj_sys::PJ_AXIS_DESCRIPTION> = Vec::with_capacity(axis_count);
        for a in axis {
            axis_vec.push(proj_sys::PJ_AXIS_DESCRIPTION {
                name: a.name.as_ptr().cast_mut(),
                abbreviation: a.abbreviation.as_ptr().cast_mut(),
                direction: a.direction.as_ptr().cast_mut(),
                unit_name: a.unit_name.as_ptr().cast_mut(),
                unit_conv_factor: a.unit_conv_factor,
                unit_type: a.unit_type.into(),
            });
        }
        let ptr = unsafe {
            proj_sys::proj_create_cs(
                self.ptr,
                coordinate_system_type.into(),
                axis_count as i32,
                axis_vec.as_ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_cartesian_2D_cs>
    pub fn create_cartesian_2d_cs(
        &self,
        ellipsoidal_cs_2d_type: CartesianCs2dType,
        unit_name: Option<&str>,
        unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let unit_name =
            CString::new(unit_name.unwrap_or_default()).expect("Error creating CString");
        let ptr = unsafe {
            proj_sys::proj_create_cartesian_2D_cs(
                self.ptr,
                ellipsoidal_cs_2d_type.into(),
                unit_name.as_ptr(),
                unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_ellipsoidal_2D_cs>
    pub fn create_ellipsoidal_2d_cs(
        &self,
        ellipsoidal_cs_2d_type: EllipsoidalCs2dType,
        unit_name: Option<&str>,
        unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let unit_name =
            CString::new(unit_name.unwrap_or_default()).expect("Error creating CString");
        let ptr = unsafe {
            proj_sys::proj_create_ellipsoidal_2D_cs(
                self.ptr,
                ellipsoidal_cs_2d_type.into(),
                unit_name.as_ptr(),
                unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_ellipsoidal_3D_cs>
    pub fn create_ellipsoidal_3d_cs(
        &self,
        ellipsoidal_cs_3d_type: EllipsoidalCs3dType,
        horizontal_angular_unit_name: Option<&str>,
        horizontal_angular_unit_conv_factor: f64,
        vertical_linear_unit_name: Option<&str>,
        vertical_linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let horizontal_angular_unit_name =
            CString::new(horizontal_angular_unit_name.unwrap_or_default())
                .expect("Error creating CString");
        let vertical_linear_unit_name = CString::new(vertical_linear_unit_name.unwrap_or_default())
            .expect("Error creating CString");
        let ptr = unsafe {
            proj_sys::proj_create_ellipsoidal_3D_cs(
                self.ptr,
                ellipsoidal_cs_3d_type.into(),
                horizontal_angular_unit_name.as_ptr(),
                horizontal_angular_unit_conv_factor,
                vertical_linear_unit_name.as_ptr(),
                vertical_linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_query_geodetic_crs_from_datum>
    fn _query_geodetic_crs_from_datum(
        &self,
        crs_auth_name: Option<&str>,
        datum_auth_name: &str,
        datum_code: &str,
        crs_type: Option<&str>,
    ) {
        let crs_auth_name =
            CString::new(crs_auth_name.unwrap_or_default()).expect("Error creating CString");
        let datum_auth_name = CString::new(datum_auth_name).expect("Error creating CString");
        let datum_code = CString::new(datum_code).expect("Error creating CString");
        let crs_type = CString::new(crs_type.unwrap_or_default()).expect("Error creating CString");
        let _ = unsafe {
            proj_sys::proj_query_geodetic_crs_from_datum(
                self.ptr,
                crs_auth_name.as_ptr(),
                datum_auth_name.as_ptr(),
                datum_code.as_ptr(),
                crs_type.as_ptr(),
            )
        };
        unimplemented!()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_geographic_crs>
    pub fn create_geographic_crs(
        &self,
        crs_name: Option<&str>,
        datum_name: Option<&str>,
        ellps_name: Option<&str>,
        semi_major_metre: f64,
        inv_flattening: f64,
        prime_meridian_name: Option<&str>,
        prime_meridian_offset: f64,
        pm_angular_units: Option<&str>,
        pm_units_conv: f64,
        ellipsoidal_cs: &Proj,
    ) -> miette::Result<Proj> {
        let crs_name = CString::new(crs_name.unwrap_or_default()).expect("Error creating CString");
        let datum_name =
            CString::new(datum_name.unwrap_or_default()).expect("Error creating CString");
        let ellps_name =
            CString::new(ellps_name.unwrap_or_default()).expect("Error creating CString");
        let prime_meridian_name =
            CString::new(prime_meridian_name.unwrap_or_default()).expect("Error creating CString");
        let pm_angular_units =
            CString::new(pm_angular_units.unwrap_or_default()).expect("Error creating CString");
        let ptr = unsafe {
            proj_sys::proj_create_geographic_crs(
                self.ptr,
                crs_name.as_ptr(),
                datum_name.as_ptr(),
                ellps_name.as_ptr(),
                semi_major_metre,
                inv_flattening,
                prime_meridian_name.as_ptr(),
                prime_meridian_offset,
                pm_angular_units.as_ptr(),
                pm_units_conv,
                ellipsoidal_cs.ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_geographic_crs_from_datum>
    pub fn create_geographic_crs_from_datum(
        &self,
        crs_name: Option<&str>,
        datum_or_datum_ensemble: &Proj,
        ellipsoidal_cs: &Proj,
    ) -> miette::Result<Proj> {
        let crs_name = CString::new(crs_name.unwrap_or_default()).expect("Error creating CString");
        let ptr = unsafe {
            proj_sys::proj_create_geographic_crs_from_datum(
                self.ptr,
                crs_name.as_ptr(),
                datum_or_datum_ensemble.ptr(),
                ellipsoidal_cs.ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// <>
    fn _create_geocentric_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_geocentric_crs_from_datum(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_derived_geographic_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _is_derived_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _alter_name(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _alter_id(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_alter_geodetic_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_alter_cs_angular_unit(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_alter_cs_linear_unit(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_alter_parameters_linear_unit(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_promote_to_3d(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_create_projected_3d_crs_from_2d(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_demote_to_2d(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_engineering_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_vertical_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_vertical_crs_ex(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_compound_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_transformation(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _convert_conversion_to_other_method(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_projected_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_create_bound_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_create_bound_vertical_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_utm(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_transverse_mercator(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_gauss_schreiber_transverse_mercator(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_transverse_mercator_south_oriented(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_two_point_equidistant(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_tunisia_mapping_grid(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_tunisia_mining_grid(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_albers_equal_area(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_lambert_conic_conformal_1sp(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_lambert_conic_conformal_1sp_variant_b(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_lambert_conic_conformal_2sp(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_lambert_conic_conformal_2sp_michigan(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_lambert_conic_conformal_2sp_belgium(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_azimuthal_equidistant(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_guam_projection(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_bonne(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_lambert_cylindrical_equal_area_spherical(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_lambert_cylindrical_equal_area(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_cassini_soldner(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_equidistant_conic(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_eckert_i(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_eckert_ii(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_eckert_iii(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_eckert_iv(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_eckert_v(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_eckert_vi(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_equidistant_cylindrical(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_equidistant_cylindrical_spherical(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_gall(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_goode_homolosine(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_interrupted_goode_homolosine(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_geostationary_satellite_sweep_x(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_geostationary_satellite_sweep_y(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_gnomonic(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_hotine_oblique_mercator_variant_a(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_hotine_oblique_mercator_variant_b(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_hotine_oblique_mercator_two_point_natural_origin(&self) {
        unimplemented!()
    }
    ///# References
    ///
    /// <>
    fn _create_conversion_laborde_oblique_mercator(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_international_map_world_polyconic(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_krovak_north_oriented(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_krovak(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_lambert_azimuthal_equal_area(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_miller_cylindrical(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_mercator_variant_a(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_mercator_variant_b(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_popular_visualisation_pseudo_mercator(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_mollweide(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_new_zealand_mapping_grid(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_oblique_stereographic(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_orthographic(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_local_orthographic(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_american_polyconic(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_polar_stereographic_variant_a(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_polar_stereographic_variant_b(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_robinson(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_sinusoidal(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_stereographic(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_wagner_i(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_wagner_ii(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_wagner_iii(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_wagner_iv(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_wagner_v(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_wagner_vi(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_wagner_vii(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_quadrilateralized_spherical_cube(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_spherical_cross_track_height(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_equal_earth(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_vertical_perspective(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_pole_rotation_grib_convention(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_conversion_pole_rotation_netcdf_cf_convention(&self) { unimplemented!() }
}
/// # ISO-19111 Base functions
impl Proj<'_> {
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_type>
    pub fn get_type(&self) -> miette::Result<ProjType> {
        let result = unsafe { proj_sys::proj_get_type(self.ptr()) };
        ProjType::try_from(result).into_diagnostic()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_deprecated>
    pub fn is_deprecated(&self) -> bool { unsafe { proj_sys::proj_is_deprecated(self.ptr()) != 0 } }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_equivalent_to>
    pub fn is_equivalent_to(&self, other: &Proj, criterion: ComparisonCriterion) -> bool {
        unsafe { proj_sys::proj_is_equivalent_to(self.ptr(), other.ptr(), criterion.into()) != 0 }
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_equivalent_to_with_ctx>
    pub fn is_equivalent_to_with_ctx(&self, other: &Proj, criterion: ComparisonCriterion) -> bool {
        unsafe {
            proj_sys::proj_is_equivalent_to_with_ctx(
                self.ctx.ptr,
                self.ptr(),
                other.ptr(),
                criterion.into(),
            ) != 0
        }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_crs>
    pub fn is_crs(&self) -> bool { unsafe { proj_sys::proj_is_crs(self.ptr()) != 0 } }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_name>
    pub fn get_name(&self) -> String {
        crate::cstr_to_string(unsafe { proj_sys::proj_get_name(self.ptr()) }).unwrap_or_default()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_id_auth_name>
    pub fn get_id_auth_name(&self, index: u16) -> Option<String> {
        crate::cstr_to_string(unsafe { proj_sys::proj_get_id_auth_name(self.ptr(), index as i32) })
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_id_code>
    pub fn get_id_code(&self, index: u16) -> Option<String> {
        crate::cstr_to_string(unsafe { proj_sys::proj_get_id_code(self.ptr(), index as i32) })
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_remarks>
    pub fn get_remarks(&self) -> String {
        crate::cstr_to_string(unsafe { proj_sys::proj_get_remarks(self.ptr()) }).unwrap_or_default()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_domain_count>
    pub fn get_domain_count(&self) -> miette::Result<u32> {
        let count = unsafe { proj_sys::proj_get_domain_count(self.ptr()) };
        if count == 0 {
            miette::bail!("get_domain_count error.")
        };
        Ok(count as u32)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_scope>
    pub fn get_scope(&self) -> Option<String> {
        crate::cstr_to_string(unsafe { proj_sys::proj_get_scope(self.ptr()) })
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_scope_ex>
    pub fn get_scope_ex(&self, domain_idx: u16) -> Option<String> {
        crate::cstr_to_string(unsafe { proj_sys::proj_get_scope_ex(self.ptr(), domain_idx as i32) })
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_area_of_use>
    pub fn get_area_of_use(&self) -> miette::Result<Option<AreaOfUse>> {
        let mut area_name: *const std::ffi::c_char = std::ptr::null();
        let mut west_lon_degree = f64::NAN;
        let mut south_lat_degree = f64::NAN;
        let mut east_lon_degree = f64::NAN;
        let mut north_lat_degree = f64::NAN;
        let result = unsafe {
            proj_sys::proj_get_area_of_use(
                self.ctx.ptr,
                self.ptr(),
                &mut west_lon_degree,
                &mut south_lat_degree,
                &mut east_lon_degree,
                &mut north_lat_degree,
                &mut area_name,
            )
        };
        if west_lon_degree == -1000.0
            || south_lat_degree == -1000.0
            || east_lon_degree == -1000.0
            || north_lat_degree == -1000.0
        {
            return Ok(None);
        }
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(Some(AreaOfUse::new(
            cstr_to_string(area_name).unwrap(),
            west_lon_degree,
            south_lat_degree,
            east_lon_degree,
            north_lat_degree,
        )))
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_area_of_use_ex>
    pub fn get_area_of_use_ex(&self, domain_idx: u16) -> miette::Result<Option<AreaOfUse>> {
        let mut area_name: *const std::ffi::c_char = std::ptr::null();
        let mut west_lon_degree = f64::NAN;
        let mut south_lat_degree = f64::NAN;
        let mut east_lon_degree = f64::NAN;
        let mut north_lat_degree = f64::NAN;
        let result = unsafe {
            proj_sys::proj_get_area_of_use_ex(
                self.ctx.ptr,
                self.ptr(),
                domain_idx as i32,
                &mut west_lon_degree,
                &mut south_lat_degree,
                &mut east_lon_degree,
                &mut north_lat_degree,
                &mut area_name,
            )
        };
        if west_lon_degree == -1000.0
            || south_lat_degree == -1000.0
            || east_lon_degree == -1000.0
            || north_lat_degree == -1000.0
        {
            return Ok(None);
        }
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(Some(AreaOfUse::new(
            cstr_to_string(area_name).unwrap(),
            west_lon_degree,
            south_lat_degree,
            east_lon_degree,
            north_lat_degree,
        )))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_as_wkt>
    pub fn as_wkt(
        &self,
        wkt_type: WktType,
        multiline: Option<bool>,
        indentation_width: Option<usize>,
        output_axis: Option<bool>,
        strict: Option<bool>,
        allow_ellipsoidal_height_as_vertical_crs: Option<bool>,
        allow_linunit_node: Option<bool>,
    ) -> miette::Result<String> {
        let mut options = crate::ProjOptions::new(6);
        options
            .push_optional(multiline, "MULTILINE", OPTION_YES)
            .push_optional(indentation_width, "INDENTATION_WIDTH", "4")
            .push_optional(output_axis, "OUTPUT_AXIS", "AUTO")
            .push_optional(strict, "STRICT", OPTION_YES)
            .push_optional(
                allow_ellipsoidal_height_as_vertical_crs,
                "ALLOW_ELLIPSOIDAL_HEIGHT_AS_VERTICAL_CRS",
                OPTION_NO,
            )
            .push_optional(allow_linunit_node, "ALLOW_LINUNIT_NODE", OPTION_YES);
        let ptrs = options.vec_ptr();
        let result = cstr_to_string(unsafe {
            proj_sys::proj_as_wkt(self.ctx.ptr, self.ptr(), wkt_type.into(), ptrs.as_ptr())
        });
        check_result!(self);
        Ok(result.expect("Error"))
    }

    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_as_proj_string>
    pub fn as_proj_string(
        &self,
        string_type: ProjStringType,
        multiline: Option<bool>,
        indentation_width: Option<usize>,
        max_line_length: Option<usize>,
    ) -> miette::Result<String> {
        let mut options = crate::ProjOptions::new(6);
        options
            .push_optional(multiline, "MULTILINE", OPTION_NO)
            .push_optional(indentation_width, "INDENTATION_WIDTH", "2")
            .push_optional(max_line_length, "MAX_LINE_LENGTH", "80");

        let ptrs = options.vec_ptr();
        let result = cstr_to_string(unsafe {
            proj_sys::proj_as_proj_string(
                self.ctx.ptr,
                self.ptr(),
                string_type.into(),
                ptrs.as_ptr(),
            )
        });
        check_result!(self);
        Ok(result.expect("Error"))
    }

    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_as_projjson>
    pub fn as_projjson(
        &self,
        multiline: Option<bool>,
        indentation_width: Option<usize>,
        schema: Option<&str>,
    ) -> miette::Result<String> {
        let mut options = crate::ProjOptions::new(6);
        options
            .push_optional(multiline, "MULTILINE", OPTION_YES)
            .push_optional(indentation_width, "INDENTATION_WIDTH", "2")
            .push_optional(schema, "SCHEMA", "");

        let ptrs = options.vec_ptr();
        let result = cstr_to_string(unsafe {
            proj_sys::proj_as_projjson(self.ctx.ptr, self.ptr(), ptrs.as_ptr())
        });
        check_result!(self);
        Ok(result.expect("Error"))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_source_crs>
    pub fn get_source_crs(&self) -> Option<Proj<'_>> {
        let out_ptr = unsafe { proj_sys::proj_get_source_crs(self.ctx.ptr, self.ptr()) };
        if out_ptr.is_null() {
            return None;
        }
        Some(Self::new(self.ctx, out_ptr).unwrap())
    }

    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_target_crs>
    pub fn get_target_crs(&self) -> Option<Proj<'_>> {
        let out_ptr = unsafe { proj_sys::proj_get_target_crs(self.ctx.ptr, self.ptr()) };
        if out_ptr.is_null() {
            return None;
        }
        Some(Self::new(self.ctx, out_ptr).unwrap())
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_identify>
    fn _identify(&self) { unimplemented!() }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_is_derived>
    pub fn crs_is_derived(&self) -> bool {
        unsafe { proj_sys::proj_crs_is_derived(self.ctx.ptr, self.ptr()) != 0 }
    }

    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_geodetic_crs>
    pub fn crs_get_geodetic_crs(&self) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_crs_get_geodetic_crs(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_horizontal_datum>
    pub fn crs_get_horizontal_datum(&self) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_crs_get_horizontal_datum(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_sub_crs>
    pub fn crs_get_sub_crs(&self, index: u16) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_crs_get_sub_crs(self.ctx.ptr, self.ptr(), index as i32) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_datum>
    pub fn crs_get_datum(&self) -> miette::Result<Option<Proj>> {
        let ptr = unsafe { proj_sys::proj_crs_get_datum(self.ctx.ptr, self.ptr()) };
        check_result!(self);
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(crate::Proj::new(self.ctx, ptr).unwrap()))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_datum_ensemble>
    pub fn crs_get_datum_ensemble(&self) -> miette::Result<Option<Proj>> {
        let ptr = unsafe { proj_sys::proj_crs_get_datum_ensemble(self.ctx.ptr, self.ptr()) };
        check_result!(self);
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(crate::Proj::new(self.ctx, ptr).unwrap()))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_datum_forced>
    pub fn crs_get_datum_forced(&self) -> miette::Result<Option<Proj>> {
        let ptr = unsafe { proj_sys::proj_crs_get_datum_forced(self.ctx.ptr, self.ptr()) };
        check_result!(self);
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(crate::Proj::new(self.ctx, ptr).unwrap()))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_has_point_motion_operation>
    pub fn crs_has_point_motion_operation(&self) -> bool {
        unsafe { proj_sys::proj_crs_has_point_motion_operation(self.ctx.ptr, self.ptr()) != 0 }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_datum_ensemble_get_member_count>
    pub fn datum_ensemble_get_member_count(&self) -> u16 {
        unsafe { proj_sys::proj_datum_ensemble_get_member_count(self.ctx.ptr, self.ptr()) as u16 }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_datum_ensemble_get_accuracy>
    pub fn datum_ensemble_get_accuracy(&self) -> miette::Result<f64> {
        let result =
            unsafe { proj_sys::proj_datum_ensemble_get_accuracy(self.ctx.ptr, self.ptr()) };
        if result < 0.0 {
            miette::bail!("Error");
        }
        Ok(result)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_datum_ensemble_get_member>
    pub fn datum_ensemble_get_member(&self, member_index: u16) -> miette::Result<Option<Proj>> {
        let ptr = unsafe {
            proj_sys::proj_datum_ensemble_get_member(self.ctx.ptr, self.ptr(), member_index as i32)
        };
        check_result!(self);
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(crate::Proj::new(self.ctx, ptr).unwrap()))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_dynamic_datum_get_frame_reference_epoch>
    pub fn dynamic_datum_get_frame_reference_epoch(&self) -> miette::Result<f64> {
        let result = unsafe {
            proj_sys::proj_dynamic_datum_get_frame_reference_epoch(self.ctx.ptr, self.ptr())
        };
        if result == -1.0 {
            miette::bail!("Error");
        }
        Ok(result)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_coordinate_system>
    pub fn crs_get_coordinate_system(&self) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_crs_get_coordinate_system(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_cs_get_type>
    pub fn cs_get_type(&self) -> miette::Result<CoordinateSystemType> {
        let cs_type = CoordinateSystemType::try_from(unsafe {
            proj_sys::proj_cs_get_type(self.ctx.ptr, self.ptr())
        })
        .into_diagnostic()?;
        if cs_type == CoordinateSystemType::Unknown {
            miette::bail!("Unknown coordinate system.");
        }
        Ok(cs_type)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_cs_get_axis_count>
    pub fn cs_get_axis_count(&self) -> miette::Result<u16> {
        let count = unsafe { proj_sys::proj_cs_get_axis_count(self.ctx.ptr, self.ptr()) };
        if count == -1 {
            miette::bail!("Error");
        }
        Ok(count as u16)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_cs_get_axis_info>
    pub fn cs_get_axis_info(&self, index: u16) -> miette::Result<AxisInfo> {
        let mut name: *const std::ffi::c_char = std::ptr::null();
        let mut abbrev: *const std::ffi::c_char = std::ptr::null();
        let mut direction: *const std::ffi::c_char = std::ptr::null();

        let mut unit_conv_factor = f64::NAN;
        let mut unit_name: *const std::ffi::c_char = std::ptr::null();
        let mut unit_auth_name: *const std::ffi::c_char = std::ptr::null();
        let mut unit_code: *const std::ffi::c_char = std::ptr::null();
        let result = unsafe {
            proj_sys::proj_cs_get_axis_info(
                self.ctx.ptr,
                self.ptr(),
                index as i32,
                &mut name,
                &mut abbrev,
                &mut direction,
                &mut unit_conv_factor,
                &mut unit_name,
                &mut unit_auth_name,
                &mut unit_code,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(AxisInfo::new(
            cstr_to_string(name).unwrap(),
            cstr_to_string(abbrev).unwrap(),
            cstr_to_string(direction).unwrap().as_str().try_into()?,
            unit_conv_factor,
            cstr_to_string(unit_name).unwrap(),
            cstr_to_string(unit_auth_name).unwrap(),
            cstr_to_string(unit_code).unwrap(),
        ))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_ellipsoid>
    pub fn get_ellipsoid(&self) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_get_ellipsoid(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_ellipsoid_get_parameters>
    pub fn ellipsoid_get_parameters(&self) -> miette::Result<EllipsoidParameters> {
        let mut semi_major_metre = f64::NAN;
        let mut semi_minor_metre = f64::NAN;
        let mut is_semi_minor_computed = i32::default();
        let mut inv_flattening = f64::NAN;
        let result = unsafe {
            proj_sys::proj_ellipsoid_get_parameters(
                self.ctx.ptr,
                self.ptr(),
                &mut semi_major_metre,
                &mut semi_minor_metre,
                &mut is_semi_minor_computed,
                &mut inv_flattening,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(EllipsoidParameters::new(
            semi_major_metre,
            semi_minor_metre,
            is_semi_minor_computed != 0,
            inv_flattening,
        ))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_celestial_body_name>
    pub fn get_celestial_body_name(&self) -> Option<String> {
        cstr_to_string(unsafe { proj_sys::proj_get_celestial_body_name(self.ctx.ptr, self.ptr()) })
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_prime_meridian>
    pub fn get_prime_meridian(&self) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_get_prime_meridian(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_prime_meridian_get_parameters>
    pub fn prime_meridian_get_parameters(&self) -> miette::Result<PrimeMeridianParameters> {
        let mut longitude = f64::NAN;
        let mut unit_conv_factor = f64::NAN;
        let mut unit_name: *const std::ffi::c_char = std::ptr::null();

        let result = unsafe {
            proj_sys::proj_prime_meridian_get_parameters(
                self.ctx.ptr,
                self.ptr(),
                &mut longitude,
                &mut unit_conv_factor,
                &mut unit_name,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(PrimeMeridianParameters::new(
            longitude,
            unit_conv_factor,
            cstr_to_string(unit_name).unwrap_or_default(),
        ))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_coordoperation>
    pub fn crs_get_coordoperation(&self) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_crs_get_coordoperation(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_method_info>
    pub fn coordoperation_get_method_info(&self) -> miette::Result<CoordOperationMethodInfo> {
        let mut method_name: *const std::ffi::c_char = std::ptr::null();
        let mut method_auth_name: *const std::ffi::c_char = std::ptr::null();
        let mut method_code: *const std::ffi::c_char = std::ptr::null();

        let result = unsafe {
            proj_sys::proj_coordoperation_get_method_info(
                self.ctx.ptr,
                self.ptr(),
                &mut method_name,
                &mut method_auth_name,
                &mut method_code,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(CoordOperationMethodInfo::new(
            cstr_to_string(method_name).unwrap_or_default(),
            cstr_to_string(method_auth_name).unwrap_or_default(),
            cstr_to_string(method_code).unwrap_or_default(),
        ))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_is_instantiable>
    pub fn coordoperation_is_instantiable(&self) -> bool {
        unsafe { proj_sys::proj_coordoperation_is_instantiable(self.ctx.ptr, self.ptr()) != 0 }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_has_ballpark_transformation>
    pub fn coordoperation_has_ballpark_transformation(&self) -> bool {
        unsafe {
            proj_sys::proj_coordoperation_has_ballpark_transformation(self.ctx.ptr, self.ptr()) != 0
        }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_requires_per_coordinate_input_time>
    pub fn coordoperation_requires_per_coordinate_input_time(&self) -> bool {
        unsafe {
            proj_sys::proj_coordoperation_requires_per_coordinate_input_time(
                self.ctx.ptr,
                self.ptr(),
            ) != 0
        }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_param_count>
    pub fn coordoperation_get_param_count(&self) -> u16 {
        unsafe { proj_sys::proj_coordoperation_get_param_count(self.ctx.ptr, self.ptr()) as u16 }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_param_index>
    pub fn coordoperation_get_param_index(&self, name: &str) -> miette::Result<u16> {
        let result = unsafe {
            proj_sys::proj_coordoperation_get_param_index(
                self.ctx.ptr,
                self.ptr(),
                CString::new(name).expect("Error creating CString").as_ptr(),
            )
        };
        if result == -1 {
            miette::bail!("Error");
        }
        Ok(result as u16)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_param>
    pub fn coordoperation_get_param(&self, index: u16) -> miette::Result<CoordOperationParam> {
        let mut name: *const std::ffi::c_char = std::ptr::null();
        let mut auth_name: *const std::ffi::c_char = std::ptr::null();
        let mut code: *const std::ffi::c_char = std::ptr::null();
        let mut value = f64::NAN;
        let mut value_string: *const std::ffi::c_char = std::ptr::null();
        let mut unit_conv_factor = f64::NAN;
        let mut unit_name: *const std::ffi::c_char = std::ptr::null();
        let mut unit_auth_name: *const std::ffi::c_char = std::ptr::null();
        let mut unit_code: *const std::ffi::c_char = std::ptr::null();
        let mut unit_category: *const std::ffi::c_char = std::ptr::null();
        let result = unsafe {
            proj_sys::proj_coordoperation_get_param(
                self.ctx.ptr,
                self.ptr(),
                index as i32,
                &mut name,
                &mut auth_name,
                &mut code,
                &mut value,
                &mut value_string,
                &mut unit_conv_factor,
                &mut unit_name,
                &mut unit_auth_name,
                &mut unit_code,
                &mut unit_category,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }

        Ok(CoordOperationParam::new(
            cstr_to_string(name).unwrap_or_default(),
            cstr_to_string(auth_name).unwrap_or_default(),
            cstr_to_string(code).unwrap_or_default(),
            value,
            cstr_to_string(value_string).unwrap_or_default(),
            unit_conv_factor,
            cstr_to_string(unit_name).unwrap_or_default(),
            cstr_to_string(unit_auth_name).unwrap_or_default(),
            cstr_to_string(unit_code).unwrap_or_default(),
            UnitCategory::try_from(unsafe { CString::from_raw(unit_category.cast_mut()) })?,
        ))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_grid_used_count>
    pub fn coordoperation_get_grid_used_count(&self) -> u16 {
        unsafe {
            proj_sys::proj_coordoperation_get_grid_used_count(self.ctx.ptr, self.ptr()) as u16
        }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_grid_used>
    pub fn coordoperation_get_grid_used(
        &self,
        index: u16,
    ) -> miette::Result<CoordOperationGridUsed> {
        let mut short_name: *const std::ffi::c_char = std::ptr::null();
        let mut full_name: *const std::ffi::c_char = std::ptr::null();
        let mut package_name: *const std::ffi::c_char = std::ptr::null();
        let mut url: *const std::ffi::c_char = std::ptr::null();
        let mut direct_download = i32::default();
        let mut open_license = i32::default();
        let mut available = i32::default();

        let result = unsafe {
            proj_sys::proj_coordoperation_get_grid_used(
                self.ctx.ptr,
                self.ptr(),
                index as i32,
                &mut short_name,
                &mut full_name,
                &mut package_name,
                &mut url,
                &mut direct_download,
                &mut open_license,
                &mut available,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(CoordOperationGridUsed::new(
            cstr_to_string(short_name).unwrap_or_default(),
            cstr_to_string(full_name).unwrap_or_default(),
            cstr_to_string(package_name).unwrap_or_default(),
            cstr_to_string(url).unwrap_or_default(),
            direct_download != 0,
            open_license != 0,
            available != 0,
        ))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_accuracy>
    pub fn coordoperation_get_accuracy(&self) -> miette::Result<f64> {
        let result =
            unsafe { proj_sys::proj_coordoperation_get_accuracy(self.ctx.ptr, self.ptr()) };
        if result < 0.0 {
            miette::bail!("Error");
        }
        Ok(result)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_towgs84_values>
    pub fn coordoperation_get_towgs84_values(&self) -> [f64; 7] {
        let mut to_wgs84 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        unsafe {
            proj_sys::proj_coordoperation_get_towgs84_values(
                self.ctx.ptr,
                self.ptr(),
                to_wgs84.as_mut_ptr(),
                7,
                1,
            )
        };
        to_wgs84
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_create_inverse>
    pub fn coordoperation_create_inverse(&self) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_coordoperation_create_inverse(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_concatoperation_get_step_count>
    pub fn concatoperation_get_step_count(&self) -> miette::Result<u16> {
        let result =
            unsafe { proj_sys::proj_concatoperation_get_step_count(self.ctx.ptr, self.ptr()) };
        if result <= 0 {
            miette::bail!("Error");
        }
        Ok(result as u16)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_concatoperation_get_step>
    pub fn concatoperation_get_step(&self, index: u16) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_concatoperation_get_step(self.ctx.ptr, self.ptr(), index as i32)
        };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordinate_metadata_create>
    pub fn coordinate_metadata_create(&self, epoch: f64) -> miette::Result<Proj> {
        let ptr =
            unsafe { proj_sys::proj_coordinate_metadata_create(self.ctx.ptr, self.ptr(), epoch) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordinate_metadata_get_epoch>
    pub fn coordinate_metadata_get_epoch(&self) -> f64 {
        unsafe { proj_sys::proj_coordinate_metadata_get_epoch(self.ctx.ptr, self.ptr()) }
    }
}
/// # ISO-19111 Advanced functions
///
/// * <https://proj.org/en/stable/development/reference/functions.html#advanced-functions>
impl Proj<'_> {
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_bound_crs_to_WGS84>
    pub fn crs_create_bound_crs_to_wgs84(
        &self,
        allow_intermediate_crs: Option<AllowIntermediateCrs>,
    ) -> miette::Result<Proj> {
        let mut options = ProjOptions::new(1);
        options.push_optional_pass(allow_intermediate_crs, "ALLOW_INTERMEDIATE_CRS");
        let vec_ptr = options.vec_ptr();
        let ptr = unsafe {
            proj_sys::proj_crs_create_bound_crs_to_WGS84(self.ctx.ptr, self.ptr(), vec_ptr.as_ptr())
        };
        crate::Proj::new(self.ctx, ptr)
    }
}
impl Clone for Proj<'_> {
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_clone>
    fn clone(&self) -> Self {
        let ptr = unsafe { proj_sys::proj_clone(self.ctx.ptr, self.ptr()) };
        let proj = Proj::new(self.ctx, ptr).unwrap();
        proj
    }
}
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_string_list_destroy>
fn string_list_destroy(ptr: *mut *mut i8) {
    unsafe {
        proj_sys::proj_string_list_destroy(ptr);
    }
}
///# References
///
/// <>
fn _proj_int_list_destroy() { unimplemented!() }
///# References
///
/// <>
fn _proj_celestial_body_list_destroy() { unimplemented!() }
///# References
///
/// <>
fn _proj_get_crs_list_parameters_create() { unimplemented!() }
///# References
///
/// <>
fn _proj_get_crs_list_parameters_destroy() { unimplemented!() }
///# References
///
/// <>
fn _proj_crs_info_list_destroy() { unimplemented!() }
///# References
///
/// <>
fn _proj_unit_list_destroy() { unimplemented!() }
///# References
///
/// <>
fn _proj_insert_object_session_create() { unimplemented!() }
///# References
///
/// <>
fn _proj_string_destroy() { unimplemented!() }
///# References
///
/// <>
fn _proj_operation_factory_context_destroy() { unimplemented!() }
/// # References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_get_count>
fn _proj_list_get_count() { unimplemented!() }
///# References
///
/// <>
fn _proj_list_destroy() { unimplemented!() }

#[cfg(test)]
mod test_context_basic {
    use super::*;
    #[test]
    fn test_set_database_path() -> miette::Result<()> {
        let _ = crate::new_test_ctx()?;
        Ok(())
    }
    #[test]
    fn test_get_database_path() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let db_path = ctx.get_database_path();
        assert!(db_path.to_string_lossy().to_string().contains(".pixi"));
        Ok(())
    }
    #[test]
    fn test_get_database_metadata() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::DatabaseLayoutVersionMajor)
            .unwrap();
        assert_eq!(data, "1");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::DatabaseLayoutVersionMinor)
            .unwrap();
        assert_eq!(data, "5");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::EpsgVersion)
            .unwrap();
        assert_eq!(data, "v12.004");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::EpsgDate)
            .unwrap();
        assert_eq!(data, "2025-03-02");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::EsriVersion)
            .unwrap();
        assert_eq!(data, "ArcGIS Pro 3.4");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::EsriDate)
            .unwrap();
        assert_eq!(data, "2024-11-04");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::IgnfSource)
            .unwrap();
        assert_eq!(
            data,
            "https://raw.githubusercontent.com/rouault/proj-resources/master/IGNF.v3.1.0.xml"
        );
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::IgnfVersion)
            .unwrap();
        assert_eq!(data, "3.1.0");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::IgnfDate)
            .unwrap();
        assert_eq!(data, "2019-05-24");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::NkgSource)
            .unwrap();
        assert_eq!(
            data,
            "https://github.com/NordicGeodesy/NordicTransformations"
        );
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::NkgVersion)
            .unwrap();
        assert_eq!(data, "1.0.w");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::NkgDate)
            .unwrap();
        assert_eq!(data, "2025-02-13");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::ProjVersion)
            .unwrap();
        assert_eq!(data, "9.6.0");
        let data = ctx.get_database_metadata(DatabaseMetadataKey::ProjDataVersion);
        assert!(data.is_none());

        Ok(())
    }
    #[test]
    fn test_get_database_structure() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let structure = ctx.get_database_structure()?;
        println!("{}", structure.first().unwrap());
        assert_eq!(
            structure.first().unwrap(),
            "CREATE TABLE metadata(\n    key TEXT NOT NULL PRIMARY KEY CHECK (length(key) >= 1),\n    value TEXT NOT NULL\n) WITHOUT ROWID;"
        );
        Ok(())
    }

    #[test]
    fn test_guess_wkt_dialect() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        let dialect = ctx.guess_wkt_dialect(&wkt)?;
        assert_eq!(dialect, GuessedWktDialect::Wkt2_2019);
        Ok(())
    }
    #[test]
    fn test_create_from_wkt() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        assert!(ctx.create_from_wkt("invalid wkt", None, None).is_err());
        ctx.create_from_wkt("ELLIPSOID[\"WGS 84\",6378137,298.257223563,\n    LENGTHUNIT[\"metre\",1],\n    ID[\"EPSG\",7030]]", None, None)?;
        Ok(())
    }
    #[test]
    fn test_create_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("32631"));
        Ok(())
    }
}
#[cfg(test)]
mod test_context_advanced {
    use super::*;
    #[test]
    fn test_create_cs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> = ctx.create_cs(
            CoordinateSystemType::Cartesian,
            &[
                AxisDescription::new(
                    Some("Longitude"),
                    Some("lon"),
                    AxisDirection::East,
                    Some("Degree"),
                    1.0,
                    UnitType::Angular,
                ),
                AxisDescription::new(
                    Some("Latitude"),
                    Some("lat"),
                    AxisDirection::North,
                    Some("Degree"),
                    1.0,
                    UnitType::Angular,
                ),
            ],
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("9122"));
        Ok(())
    }
    #[test]
    fn test_create_cartesian_2d_cs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> =
            ctx.create_cartesian_2d_cs(CartesianCs2dType::EastingNorthing, Some("Degree"), 1.0)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("LENGTHUNIT"));
        Ok(())
    }
    #[test]
    fn test_create_ellipsoidal_2d_cs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> = ctx.create_ellipsoidal_2d_cs(
            EllipsoidalCs2dType::LatitudeLongitude,
            Some("Degree"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("9122"));
        Ok(())
    }
    #[test]
    fn test_create_ellipsoidal_3d_cs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> = ctx.create_ellipsoidal_3d_cs(
            EllipsoidalCs3dType::LatitudeLongitudeHeight,
            Some("Degree"),
            1.0,
            Some("Degree"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("LENGTHUNIT"));
        Ok(())
    }
    #[test]
    fn test_create_geographic_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> = ctx.create_geographic_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            1.0,
            Some("Degree"),
            1.0,
            &ctx.create_ellipsoidal_2d_cs(
                EllipsoidalCs2dType::LatitudeLongitude,
                Some("Degree"),
                1.0,
            )?,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("9122"));

        Ok(())
    }
    #[test]
    fn test_create_geographic_crs_from_datum() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> = ctx.create_geographic_crs_from_datum(
            Some("WGS 84"),
            &ctx.create("+proj=geocent +ellps=GRS80 +units=m +no_defs +type=crs")?
                .crs_get_datum()?
                .unwrap(),
            &ctx.create_ellipsoidal_2d_cs(
                EllipsoidalCs2dType::LatitudeLongitude,
                Some("Degree"),
                1.0,
            )?,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("9122"));

        Ok(())
    }
}
#[cfg(test)]
mod test_proj_basic {
    use super::*;

    #[test]
    fn test_get_type() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let t = pj.get_type()?;
        assert_eq!(t, ProjType::Geographic2dCrs);
        Ok(())
    }
    #[test]
    fn test_is_deprecated() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let deprecated = pj.is_deprecated();
        assert!(!deprecated);
        Ok(())
    }
    #[test]
    fn test_is_equivalent_to() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj1 = ctx.create("EPSG:4326")?;
        let pj2 = ctx.create("EPSG:4496")?;
        let equivalent = pj1.is_equivalent_to(&pj2, ComparisonCriterion::Equivalent);
        assert!(!equivalent);
        Ok(())
    }
    #[test]
    fn test_is_equivalent_to_with_ctx() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj1 = ctx.create("EPSG:4326")?;
        let pj2 = ctx.create("EPSG:4496")?;
        let equivalent = pj1.is_equivalent_to_with_ctx(&pj2, ComparisonCriterion::Equivalent);
        assert!(!equivalent);
        Ok(())
    }
    #[test]
    fn test_is_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let result = pj.is_crs();
        assert!(result);
        Ok(())
    }
    #[test]
    fn test_get_name() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let name = pj.get_name();
        println!("{name}");
        assert_eq!(name, "WGS 84");
        Ok(())
    }
    #[test]
    fn test_get_id_auth_name() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let id_auth_name = pj.get_id_auth_name(0).expect("No id_auth_name");
        println!("{id_auth_name}");
        assert_eq!(id_auth_name, "EPSG");
        Ok(())
    }
    #[test]
    fn test_get_id_code() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let id = pj.get_id_code(0).expect("No id_code");
        println!("{id}");
        assert_eq!(id, "4326");
        Ok(())
    }
    #[test]
    fn test_get_remarks() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let remarks = pj.get_remarks();
        println!("{remarks}");
        Ok(())
    }
    #[test]
    fn test_get_domain_count() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let count = pj.get_domain_count()?;
        assert_eq!(count, 1);
        Ok(())
    }
    #[test]
    fn test_get_scope() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let scope = pj.get_scope().expect("No scope");
        println!("{scope}");
        assert_eq!(scope, "Horizontal component of 3D system.");
        Ok(())
    }
    #[test]
    fn test_get_scope_ex() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let scope = pj.get_scope_ex(0).expect("No scope");
        println!("{scope}");
        assert_eq!(scope, "Horizontal component of 3D system.");
        Ok(())
    }
    #[test]
    pub fn test_as_wkt() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("WGS 84"));
        Ok(())
    }
    #[test]
    pub fn test_as_proj_string() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let proj_string = pj.as_proj_string(ProjStringType::Proj4, None, None, None)?;
        println!("{proj_string}");
        assert_eq!(proj_string, "+proj=longlat +datum=WGS84 +no_defs +type=crs");
        Ok(())
    }
    #[test]
    pub fn test_as_projjson() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let json = pj.as_projjson(None, None, None)?;
        println!("{json}");
        assert!(json.contains("WGS 84"));
        Ok(())
    }
    #[test]
    pub fn test_get_source_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:3857", &crate::Area::default())?;
        let target = pj.get_source_crs().unwrap();
        assert_eq!(target.get_name(), "WGS 84");
        Ok(())
    }
    #[test]
    pub fn test_get_target_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:3857", &crate::Area::default())?;
        let target = pj.get_target_crs().unwrap();
        assert_eq!(target.get_name(), "WGS 84 / Pseudo-Mercator");
        Ok(())
    }
    #[test]
    fn test_crs_is_derived() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        assert!(pj.is_crs());
        let derived = pj.crs_is_derived();
        assert!(!derived);
        Ok(())
    }
    #[test]
    fn test_crs_get_geodetic_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:3857")?;
        assert!(pj.is_crs());
        let geodetic = pj.crs_get_geodetic_crs()?;
        let wkt = geodetic.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("4326"));
        Ok(())
    }
    #[test]
    fn test_crs_get_horizontal_datum() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:3857")?;
        assert!(pj.is_crs());
        let horizontal = pj.crs_get_horizontal_datum()?;
        let wkt = horizontal.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("6326"));
        Ok(())
    }
    #[test]
    fn test_crs_get_sub_crs() -> miette::Result<()> { Ok(()) }
    #[test]
    fn test_crs_get_datum() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("+proj=geocent +ellps=GRS80 +units=m +no_defs +type=crs")?;
        assert!(pj.is_crs());
        let datum = pj.crs_get_datum()?;
        assert!(datum.is_some());
        Ok(())
    }
    #[test]
    fn test_crs_get_datum_ensemble() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4258")?;
        assert!(pj.is_crs());
        let datum = pj.crs_get_datum_ensemble()?;
        assert!(datum.is_some());
        Ok(())
    }
    #[test]
    fn test_crs_get_datum_forced() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("+proj=geocent +ellps=GRS80 +units=m +no_defs +type=crs")?;
        assert!(pj.is_crs());
        let datum = pj.crs_get_datum_forced()?;
        assert!(datum.is_some());
        Ok(())
    }
    #[test]
    fn test_crs_has_point_motion_operation() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:8255")?;
        assert!(pj.is_crs());
        let result = pj.crs_has_point_motion_operation();
        assert!(result);
        Ok(())
    }
    #[test]
    fn test_datum_ensemble_get_member_count() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4258")?;
        let datum = pj.crs_get_datum_ensemble()?.unwrap();
        let count = datum.datum_ensemble_get_member_count();
        assert_eq!(count, 12);
        Ok(())
    }
    #[test]
    fn test_datum_ensemble_get_accuracy() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4258")?;
        let datum = pj.crs_get_datum_ensemble()?.unwrap();
        let accuracy = datum.datum_ensemble_get_accuracy()?;
        assert_eq!(accuracy, 0.1);
        Ok(())
    }
    #[test]
    fn test_datum_ensemble_get_member() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4258")?;
        let datum = pj.crs_get_datum_ensemble()?.unwrap();
        let _ = datum.datum_ensemble_get_member(2)?.unwrap();
        Ok(())
    }
    #[test]
    fn test_dynamic_datum_get_frame_reference_epoch() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "1061", Category::Datum, false)?;
        let epoch = pj.dynamic_datum_get_frame_reference_epoch()?;
        assert_eq!(epoch, 2005.0);
        Ok(())
    }
    #[test]
    fn test_crs_get_coordinate_system() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let cs = pj.crs_get_coordinate_system()?;
        let wkt = cs.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("9122"));
        Ok(())
    }
    #[test]
    fn test_cs_get_type() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let cs = pj.crs_get_coordinate_system()?;
        let t = cs.cs_get_type()?;
        assert_eq!(t, CoordinateSystemType::Ellipsoidal);
        Ok(())
    }
    #[test]
    fn test_cs_get_axis_count() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let cs = pj.crs_get_coordinate_system()?;
        let count = cs.cs_get_axis_count()?;
        assert_eq!(count, 2);
        Ok(())
    }
    #[test]
    fn test_cs_get_axis_info() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let cs = pj.crs_get_coordinate_system()?;
        let info = cs.cs_get_axis_info(1)?;
        println!("{:?}", info);
        assert_eq!(info.name(), "Geodetic longitude");
        assert_eq!(info.abbrev(), "Lon");
        assert_eq!(info.direction(), &AxisDirection::East);
        assert_eq!(info.unit_conv_factor(), &0.017453292519943295);
        assert_eq!(info.unit_name(), "degree");
        assert_eq!(info.unit_auth_name(), "EPSG");
        assert_eq!(info.unit_code(), "9122");
        Ok(())
    }
    #[test]
    fn test_get_ellipsoid() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let ellps = pj.get_ellipsoid()?;
        let wkt = ellps.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("7030"));
        Ok(())
    }
    #[test]
    fn test_ellipsoid_get_parameters() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let ellps = pj.get_ellipsoid()?;
        let param = ellps.ellipsoid_get_parameters()?;
        println!("{:?}", param);
        Ok(())
    }
    #[test]
    fn test_get_celestial_body_name() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let name = pj.get_celestial_body_name().unwrap();
        assert_eq!(name, "Earth");
        Ok(())
    }
    #[test]
    fn test_get_prime_meridian() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let meridian = pj.get_prime_meridian()?;
        let wkt = meridian.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("8901"));
        Ok(())
    }
    #[test]
    fn test_prime_meridian_get_parameters() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let meridian = pj.get_prime_meridian()?;
        let params = meridian.prime_meridian_get_parameters()?;
        println!("{:?}", params);
        assert_eq!(
            format!("{:?}", params),
            "PrimeMeridianParameters { longitude: 0.0, unit_conv_factor: 0.017453292519943295, unit_name: \"degree\" }"
        );
        Ok(())
    }
    #[test]
    fn test_crs_get_coordoperation() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let wkt = op.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("16031"));
        Ok(())
    }
    #[test]
    fn test_coordoperation_get_method_info() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let info = op.coordoperation_get_method_info()?;
        println!("{:?}", info);
        assert_eq!(
            format!("{:?}", info),
            "CoordOperationMethodInfo { method_name: \"Transverse Mercator\", method_auth_name: \"EPSG\", method_code: \"9807\" }"
        );
        Ok(())
    }
    #[test]
    fn test_coordoperation_is_instantiable() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let instantiable = op.coordoperation_is_instantiable();
        assert!(instantiable);
        Ok(())
    }
    #[test]
    fn test_coordoperation_has_ballpark_transformation() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let has_ballpark_transformation = op.coordoperation_has_ballpark_transformation();
        assert!(!has_ballpark_transformation);
        Ok(())
    }
    #[test]
    fn test_coordoperation_requires_per_coordinate_input_time() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let requires_per_coordinate_input_time =
            op.coordoperation_requires_per_coordinate_input_time();
        assert!(!requires_per_coordinate_input_time);
        Ok(())
    }
    #[test]
    fn test_coordoperation_get_param_count() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let count = op.coordoperation_get_param_count();
        assert_eq!(count, 5);
        Ok(())
    }
    #[test]
    fn test_coordoperation_get_param_index() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let index = op.coordoperation_get_param_index("Longitude of natural origin")?;
        assert_eq!(index, 1);
        Ok(())
    }
    #[test]
    fn test_coordoperation_get_param() -> miette::Result<()> {
        // let ctx = crate::new_test_ctx()?;
        // let pj = ctx.create_from_database("EPSG", "1037",
        // Category::CoordinateOperation, false)?; let param =
        // pj.coordoperation_get_param(1)?; println!("{:?}", param);
        Ok(())
    }
    #[test]
    fn test_coordoperation_get_grid_used_count() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "1312", Category::CoordinateOperation, true)?;
        let count = pj.coordoperation_get_grid_used_count();
        assert_eq!(count, 1);
        Ok(())
    }

    #[test]
    fn test_coordoperation_get_grid_used() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "1312", Category::CoordinateOperation, true)?;
        let grid = pj.coordoperation_get_grid_used(0)?;
        println!("{:?}", grid);
        assert!(format!("{:?}", grid).contains("ca_nrc_ntv1_can.tif"));
        Ok(())
    }
    #[test]
    fn test_coordoperation_get_accuracy() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "8048", Category::CoordinateOperation, false)?;
        let accuracy = pj.coordoperation_get_accuracy()?;
        assert_eq!(accuracy, 0.01);
        Ok(())
    }
    #[test]
    fn test_coordoperation_get_towgs84_values() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "8048", Category::CoordinateOperation, false)?;
        let param = pj.coordoperation_get_towgs84_values();
        assert_eq!(
            param,
            [
                0.06155,
                -0.01087,
                -0.04019,
                0.03949239999999997,
                0.03272209999999997,
                0.032897899999999966,
                -0.009994000000000001
            ]
        );
        Ok(())
    }
    #[test]
    fn test_coordoperation_create_inverse() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let inversed = op.coordoperation_create_inverse()?;
        let wkt = inversed.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("16031"));
        Ok(())
    }
    #[test]
    fn test_concatoperation_get_step_count() -> miette::Result<()> {
        // let ctx = crate::new_test_ctx()?;
        // let pj = ctx.create("EPSG:4326")?;
        // let bound =
        // pj.crs_create_bound_crs_to_wgs84(Some(AllowIntermediateCrs::Never))?;
        // let op = bound.crs_get_coordoperation()?;
        // let count = op.concatoperation_get_step_count()?;
        // assert_eq!(count, 1);
        Ok(())
    }
    #[test]
    fn test_concatoperation_get_step() -> miette::Result<()> {
        // let ctx = crate::new_test_ctx()?;
        // let pj = ctx.create_from_database("EPSG", "28356",
        // Category::CoordinateOperation, false)?; let step =
        // pj.concatoperation_get_step(1)?; let wkt = step.as_wkt(
        //     WktType::Wkt2_2019,
        //     None,
        //     None,
        //     None,
        //     None,
        //     None,
        //     None,
        // )?;
        // println!("{}", wkt);
        // assert!(wkt.contains("16031"));
        Ok(())
    }
    #[test]
    fn test_coordinate_metadata_create() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let new = pj.coordinate_metadata_create(123.4)?;
        let wkt = new.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}",);
        assert!(wkt.contains("123.4"));
        Ok(())
    }
    #[test]
    fn test_coordinate_metadata_get_epoch() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let new = pj.coordinate_metadata_create(123.4)?;
        let epoch = new.coordinate_metadata_get_epoch();
        assert_eq!(epoch, 123.4);
        Ok(())
    }
}
#[cfg(test)]
mod test_proj_advanced {
    use super::*;
    #[test]
    fn test_crs_create_bound_crs_to_wgs84() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let bound = pj.crs_create_bound_crs_to_wgs84(Some(AllowIntermediateCrs::Never))?;
        let wkt = bound.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}",);
        assert!(wkt.contains("32631"));
        Ok(())
    }
}
