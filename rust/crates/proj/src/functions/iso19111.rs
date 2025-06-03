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
use std::str::FromStr;

use envoy::{CStrListToVecString, CStrToString, ToCStr};
use miette::IntoDiagnostic;

use crate::data_types::iso19111::*;
use crate::{Context, OPTION_NO, OPTION_YES, Proj, ProjOptions, check_result, pj_obj_list_to_vec};
/// # ISO-19111 Base functions
impl crate::Context {
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_autoclose_database>
    #[deprecated]
    fn _set_autoclose_database(&self) { unimplemented!("Deprecated") }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_database_path>
    pub fn set_database_path(
        &self,
        db_path: &Path,
        aux_db_paths: Option<&[PathBuf]>,
    ) -> miette::Result<&Self> {
        let aux_db_paths: Option<Vec<CString>> = aux_db_paths.map(|aux_db_paths| {
            aux_db_paths
                .iter()
                .map(|f| f.to_str().to_cstring())
                .collect()
        });

        let aux_db_paths_ptr: Option<Vec<*const i8>> =
            aux_db_paths.map(|aux_db_paths| aux_db_paths.iter().map(|f| f.as_ptr()).collect());

        let result = unsafe {
            proj_sys::proj_context_set_database_path(
                self.ptr,
                db_path.to_str().to_cstr(),
                aux_db_paths_ptr.map_or(ptr::null(), |ptr| ptr.as_ptr()),
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
            unsafe { proj_sys::proj_context_get_database_path(self.ptr) }
                .to_string()
                .unwrap_or_default(),
        )
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_database_metadata>
    pub fn get_database_metadata(&self, key: DatabaseMetadataKey) -> Option<String> {
        let key = key.as_ref().to_cstring();
        unsafe { proj_sys::proj_context_get_database_metadata(self.ptr, key.as_ptr()) }.to_string()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_database_structure>
    pub fn get_database_structure(&self) -> miette::Result<Vec<String>> {
        let ptr = unsafe { proj_sys::proj_context_get_database_structure(self.ptr, ptr::null()) };
        let out_vec = ptr.to_vec_string().unwrap();
        string_list_destroy(ptr);
        Ok(out_vec)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_guess_wkt_dialect>
    pub fn guess_wkt_dialect(&self, wkt: &str) -> miette::Result<GuessedWktDialect> {
        GuessedWktDialect::try_from(unsafe {
            proj_sys::proj_context_guess_wkt_dialect(self.ptr, wkt.to_cstr())
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
                wkt.to_cstr(),
                vec_ptr.as_ptr(),
                &mut out_warnings,
                &mut out_grammar_errors,
            )
        };
        //warning
        if let Some(warnings) = out_warnings.to_vec_string() {
            warnings.iter().for_each(|w| clerk::warn!("{w}"))
        }
        //error
        if let Some(warnings) = out_grammar_errors.to_vec_string() {
            warnings.iter().for_each(|w| clerk::error!("{w}"))
        }
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
        let ptr = unsafe {
            proj_sys::proj_create_from_database(
                self.ptr,
                auth_name.to_cstr(),
                code.to_cstr(),
                category.into(),
                use_projalternative_grid_names as i32,
                null(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_uom_get_info_from_database>
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
                auth_name.to_cstr(),
                code.to_cstr(),
                &mut name,
                &mut conv_factor,
                &mut category,
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }

        Ok(UomInfo::new(
            name.to_string().unwrap(),
            conv_factor,
            UomCategory::from_str(&category.to_string().unwrap()).into_diagnostic()?,
        ))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_get_info_from_database>
    pub fn grid_get_info_from_database(&self, grid_name: &str) -> miette::Result<GridInfoDB> {
        let mut full_name: *const std::ffi::c_char = std::ptr::null();
        let mut package_name: *const std::ffi::c_char = std::ptr::null();
        let mut url: *const std::ffi::c_char = std::ptr::null();
        let mut direct_download: i32 = i32::default();
        let mut open_license: i32 = i32::default();
        let mut available: i32 = i32::default();
        let result = unsafe {
            proj_sys::proj_grid_get_info_from_database(
                self.ptr,
                grid_name.to_cstr(),
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
            full_name.to_string().unwrap(),
            package_name.to_string().unwrap(),
            url.to_string().unwrap(),
            direct_download != 0,
            open_license != 0,
            available != 0,
        ))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_from_name>
    pub fn create_from_name(
        &self,
        auth_name: Option<&str>,
        searched_name: &str,
        types: Option<&[ProjType]>,
        approximate_match: bool,
        limit_result_count: usize,
    ) -> miette::Result<Vec<Proj>> {
        let (types, count) = types.map_or((None, 0), |types| {
            let types: Vec<u32> = types.iter().map(|f| u32::from(f.clone())).collect();
            let count = types.len();
            (Some(types), count)
        });
        let result = unsafe {
            proj_sys::proj_create_from_name(
                self.ptr,
                auth_name.to_cstr(),
                searched_name.to_cstr(),
                types.map_or(ptr::null(), |types| types.as_ptr()),
                count,
                approximate_match as i32,
                limit_result_count,
                ptr::null(),
            )
        };
        pj_obj_list_to_vec(self, result)
    }

    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_geoid_models_from_database>
    pub fn get_geoid_models_from_database(
        &self,
        auth_name: &str,
        code: &str,
    ) -> miette::Result<Vec<String>> {
        let ptr = unsafe {
            proj_sys::proj_get_geoid_models_from_database(
                self.ptr,
                auth_name.to_cstr(),
                code.to_cstr(),
                ptr::null(),
            )
        };
        if ptr.is_null() {
            miette::bail!("Error");
        }
        let out_vec = ptr.to_vec_string().unwrap();
        string_list_destroy(ptr);
        Ok(out_vec)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_authorities_from_database>
    pub fn get_authorities_from_database(&self) -> miette::Result<Vec<String>> {
        let ptr = unsafe { proj_sys::proj_get_authorities_from_database(self.ptr) };
        if ptr.is_null() {
            miette::bail!("Error");
        }
        let out_vec = ptr.to_vec_string().unwrap();
        string_list_destroy(ptr);
        Ok(out_vec)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_codes_from_database>
    pub fn get_codes_from_database(
        &self,
        auth_name: &str,
        proj_type: ProjType,
        allow_deprecated: bool,
    ) -> miette::Result<Vec<String>> {
        let ptr = unsafe {
            proj_sys::proj_get_codes_from_database(
                self.ptr,
                auth_name.to_cstr(),
                proj_type.into(),
                allow_deprecated as i32,
            )
        };
        if ptr.is_null() {
            miette::bail!("Error");
        }
        let out_vec = ptr.to_vec_string().unwrap_or_default();
        string_list_destroy(ptr);
        Ok(out_vec)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_celestial_body_list_from_database>
    pub fn get_celestial_body_list_from_database(
        &self,
        auth_name: &str,
    ) -> miette::Result<Vec<CelestialBodyInfo>> {
        let mut out_result_count = i32::default();
        let ptr = unsafe {
            proj_sys::proj_get_celestial_body_list_from_database(
                self.ptr,
                auth_name.to_cstr(),
                &mut out_result_count,
            )
        };
        if out_result_count < 1 {
            miette::bail!("Error");
        }
        let mut out_vec = Vec::new();
        for offset in 0..out_result_count {
            let current_ptr = unsafe { ptr.offset(offset as isize).as_ref().unwrap() };
            let info_ref = unsafe { current_ptr.as_ref().unwrap() };
            out_vec.push(CelestialBodyInfo::new(
                info_ref.auth_name.to_string().unwrap(),
                info_ref.name.to_string().unwrap(),
            ));
        }
        unsafe { proj_sys::proj_celestial_body_list_destroy(ptr) };
        Ok(out_vec)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_crs_info_list_from_database>
    pub fn get_crs_info_list_from_database(
        &self,
        auth_name: Option<&str>,
        params: Option<CrsListParameters>,
    ) -> miette::Result<Vec<CrsInfo>> {
        if auth_name.is_none() && params.is_none() {
            miette::bail!("At least one of `auth_name` and  `params` must be set.");
        }
        let mut out_result_count = i32::default();
        let params = if let Some(params) = params {
            let types: Vec<u32> = params
                .types()
                .to_owned()
                .iter()
                .map(|f| u32::from(f.clone()))
                .collect();
            let celestial_body_name = params.celestial_body_name().to_owned().to_cstr();
            Some(proj_sys::PROJ_CRS_LIST_PARAMETERS {
                types: types.as_ptr(),
                typesCount: params.types().len(),
                crs_area_of_use_contains_bbox: *params.west_lon_degree() as i32,
                bbox_valid: *params.bbox_valid() as i32,
                west_lon_degree: *params.west_lon_degree(),
                south_lat_degree: *params.south_lat_degree(),
                east_lon_degree: *params.east_lon_degree(),
                north_lat_degree: *params.north_lat_degree(),
                allow_deprecated: *params.allow_deprecated() as i32,
                celestial_body_name,
            })
        } else {
            None
        };

        let ptr = unsafe {
            proj_sys::proj_get_crs_info_list_from_database(
                self.ptr,
                auth_name.to_cstr(),
                params.map_or(ptr::null(), |p| &p),
                &mut out_result_count,
            )
        };
        if out_result_count < 1 {
            miette::bail!("Error");
        }
        let mut out_vec = Vec::new();
        for offset in 0..1803 {
            let current_ptr = unsafe { ptr.offset(offset as isize).as_ref().unwrap() };
            let info_ref = unsafe { current_ptr.as_ref().unwrap() };
            out_vec.push(CrsInfo::new(
                info_ref.auth_name.to_string().unwrap(),
                info_ref.code.to_string().unwrap(),
                info_ref.name.to_string().unwrap(),
                ProjType::try_from(info_ref.type_).into_diagnostic()?,
                info_ref.deprecated != 0,
                info_ref.bbox_valid != 0,
                info_ref.west_lon_degree,
                info_ref.south_lat_degree,
                info_ref.east_lon_degree,
                info_ref.north_lat_degree,
                info_ref.area_name.to_string().unwrap_or_default(),
                info_ref
                    .projection_method_name
                    .to_string()
                    .unwrap_or_default(),
                info_ref.celestial_body_name.to_string().unwrap_or_default(),
            ));
        }
        unsafe { proj_sys::proj_crs_info_list_destroy(ptr) };
        Ok(out_vec)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_units_from_database>
    pub fn get_units_from_database(
        &self,
        auth_name: &str,
        category: UnitCategory,
        allow_deprecated: bool,
    ) -> miette::Result<Vec<UnitInfo>> {
        let mut out_result_count = i32::default();
        let ptr = unsafe {
            proj_sys::proj_get_units_from_database(
                self.ptr,
                auth_name.to_cstr(),
                category.as_ref().to_cstr(),
                allow_deprecated as i32,
                &mut out_result_count,
            )
        };
        if out_result_count < 1 {
            miette::bail!("Error");
        }
        let mut out_vec = Vec::new();
        for offset in 0..out_result_count {
            let current_ptr = unsafe { ptr.offset(offset as isize).as_ref().unwrap() };
            let info_ref = unsafe { current_ptr.as_ref().unwrap() };
            out_vec.push(UnitInfo::new(
                info_ref.auth_name.to_string().unwrap(),
                info_ref.code.to_string().unwrap(),
                info_ref.name.to_string().unwrap(),
                UnitCategory::from_str(&info_ref.category.to_string().unwrap())
                    .into_diagnostic()?,
                info_ref.conv_factor,
                info_ref.code.to_string().unwrap(),
                info_ref.deprecated != 0,
            ));
        }
        unsafe { proj_sys::proj_unit_list_destroy(ptr) };
        Ok(out_vec)
    }
    ///# References
    ///
    /// <>
    fn _insert_object_session_create(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _insert_object_session_destroy(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _get_insert_statements(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _suggests_code_for(&self) { todo!() }
    ///# References
    ///
    /// <>
    fn _create_operation_factory_context(&self) { todo!() }
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
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_get>
    pub(crate) fn list_get(
        &self,
        result: *const proj_sys::PJ_OBJ_LIST,
        index: i32,
    ) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_list_get(self.ptr, result, index) };
        Proj::new(self, ptr)
    }
    ///# References
    ///
    /// <>
    fn _get_suggested_operation(&self) { todo!() }
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
        let ptr = unsafe {
            proj_sys::proj_create_cartesian_2D_cs(
                self.ptr,
                ellipsoidal_cs_2d_type.into(),
                unit_name.to_cstr(),
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
        let ptr = unsafe {
            proj_sys::proj_create_ellipsoidal_2D_cs(
                self.ptr,
                ellipsoidal_cs_2d_type.into(),
                unit_name.to_cstr(),
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
        let ptr = unsafe {
            proj_sys::proj_create_ellipsoidal_3D_cs(
                self.ptr,
                ellipsoidal_cs_3d_type.into(),
                horizontal_angular_unit_name.to_cstr(),
                horizontal_angular_unit_conv_factor,
                vertical_linear_unit_name.to_cstr(),
                vertical_linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_query_geodetic_crs_from_datum>
    pub fn query_geodetic_crs_from_datum(
        &self,
        crs_auth_name: Option<&str>,
        datum_auth_name: &str,
        datum_code: &str,
        crs_type: Option<&str>,
    ) -> miette::Result<Vec<Proj>> {
        let result = unsafe {
            proj_sys::proj_query_geodetic_crs_from_datum(
                self.ptr,
                crs_auth_name.to_cstr(),
                datum_auth_name.to_cstr(),
                datum_code.to_cstr(),
                crs_type.to_cstr(),
            )
        };
        pj_obj_list_to_vec(self, result)
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
        let ptr = unsafe {
            proj_sys::proj_create_geographic_crs(
                self.ptr,
                crs_name.to_cstr(),
                datum_name.to_cstr(),
                ellps_name.to_cstr(),
                semi_major_metre,
                inv_flattening,
                prime_meridian_name.to_cstr(),
                prime_meridian_offset,
                pm_angular_units.to_cstr(),
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
        let ptr = unsafe {
            proj_sys::proj_create_geographic_crs_from_datum(
                self.ptr,
                crs_name.to_cstr(),
                datum_or_datum_ensemble.ptr(),
                ellipsoidal_cs.ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_geocentric_crs>
    pub fn create_geocentric_crs(
        &self,
        crs_name: Option<&str>,
        datum_name: Option<&str>,
        ellps_name: Option<&str>,
        semi_major_metre: f64,
        inv_flattening: f64,
        prime_meridian_name: Option<&str>,
        prime_meridian_offset: f64,
        angular_units: Option<&str>,
        angular_units_conv: f64,
        linear_units: Option<&str>,
        linear_units_conv: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_geocentric_crs(
                self.ptr,
                crs_name.to_cstr(),
                datum_name.to_cstr(),
                ellps_name.to_cstr(),
                semi_major_metre,
                inv_flattening,
                prime_meridian_name.to_cstr(),
                prime_meridian_offset,
                angular_units.to_cstr(),
                angular_units_conv,
                linear_units.to_cstr(),
                linear_units_conv,
            )
        };
        Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_geocentric_crs_from_datum>
    pub fn create_geocentric_crs_from_datum(
        &self,
        crs_name: Option<&str>,
        datum_or_datum_ensemble: &Proj,
        linear_units: Option<&str>,
        linear_units_conv: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_geocentric_crs_from_datum(
                self.ptr,
                crs_name.to_cstr(),
                datum_or_datum_ensemble.ptr(),
                linear_units.to_cstr(),
                linear_units_conv,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_derived_geographic_crs>
    pub fn create_derived_geographic_crs(
        &self,
        crs_name: Option<&str>,
        base_geographic_crs: &Proj,
        conversion: &Proj,
        ellipsoidal_cs: &Proj,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_derived_geographic_crs(
                self.ptr,
                crs_name.to_cstr(),
                base_geographic_crs.ptr(),
                conversion.ptr(),
                ellipsoidal_cs.ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_projected_3D_crs_from_2D>
    pub fn crs_create_projected_3d_crs_from_2d(
        &self,
        crs_name: Option<&str>,
        projected_2d_crs: &Proj,
        geog_3d_crs: Option<&Proj>,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_create_projected_3D_crs_from_2D(
                self.ptr,
                crs_name.to_cstr(),
                projected_2d_crs.ptr(),
                geog_3d_crs.map_or(ptr::null(), |crs| crs.ptr()),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_engineering_crs>
    pub fn create_engineering_crs(&self, crs_name: Option<&str>) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_create_engineering_crs(self.ptr, crs_name.to_cstr()) };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_vertical_crs>
    pub fn create_vertical_crs(
        &self,
        crs_name: Option<&str>,
        datum_name: Option<&str>,
        linear_units: Option<&str>,
        linear_units_conv: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_vertical_crs(
                self.ptr,
                crs_name.to_cstr(),
                datum_name.to_cstr(),
                linear_units.to_cstr(),
                linear_units_conv,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_vertical_crs_ex>
    pub fn create_vertical_crs_ex(
        &self,
        crs_name: Option<&str>,
        datum_name: Option<&str>,
        datum_auth_name: Option<&str>,
        datum_code: Option<&str>,
        linear_units: Option<&str>,
        linear_units_conv: f64,
        geoid_model_name: Option<&str>,
        geoid_model_auth_name: Option<&str>,
        geoid_model_code: Option<&str>,
        geoid_geog_crs: Option<&Proj>,
        accuracy: Option<f64>,
    ) -> miette::Result<Proj> {
        let mut option = ProjOptions::new(1);
        option.push_optional_pass(accuracy, "ACCURACY");

        let ptr = unsafe {
            proj_sys::proj_create_vertical_crs_ex(
                self.ptr,
                crs_name.to_cstr(),
                datum_name.to_cstr(),
                datum_auth_name.to_cstr(),
                datum_code.to_cstr(),
                linear_units.to_cstr(),
                linear_units_conv,
                geoid_model_name.to_cstr(),
                geoid_model_auth_name.to_cstr(),
                geoid_model_code.to_cstr(),
                geoid_geog_crs.map_or(ptr::null(), |crs| crs.ptr()),
                option.vec_ptr().as_mut_ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_compound_crs>
    pub fn create_compound_crs(
        &self,
        crs_name: Option<&str>,
        horiz_crs: &Proj,
        vert_crs: &Proj,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_compound_crs(
                self.ptr,
                crs_name.to_cstr(),
                horiz_crs.ptr(),
                vert_crs.ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion>
    pub fn create_conversion(
        &self,
        name: Option<&str>,
        auth_name: Option<&str>,
        code: Option<&str>,
        method_name: Option<&str>,
        method_auth_name: Option<&str>,
        method_code: Option<&str>,
        params: &[ParamDescription],
    ) -> miette::Result<Proj> {
        let count = params.len();
        let params: Vec<proj_sys::PJ_PARAM_DESCRIPTION> = params
            .iter()
            .map(|p| proj_sys::PJ_PARAM_DESCRIPTION {
                name: p.name().to_cstr(),
                auth_name: p.auth_name().to_cstr(),
                code: p.code().to_cstr(),
                value: *p.value(),
                unit_name: p.unit_name().to_cstr(),
                unit_conv_factor: *p.unit_conv_factor(),
                unit_type: u32::from(*p.unit_type()),
            })
            .collect();
        let ptr = unsafe {
            proj_sys::proj_create_conversion(
                self.ptr,
                name.to_cstr(),
                auth_name.to_cstr(),
                code.to_cstr(),
                method_name.to_cstr(),
                method_auth_name.to_cstr(),
                method_code.to_cstr(),
                count as i32,
                params.as_ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_transformation>
    pub fn create_transformation(
        &self,
        name: Option<&str>,
        auth_name: Option<&str>,
        code: Option<&str>,
        source_crs: Option<&Proj>,
        target_crs: Option<&Proj>,
        interpolation_crs: Option<&Proj>,
        method_name: Option<&str>,
        method_auth_name: Option<&str>,
        method_code: Option<&str>,
        params: &[ParamDescription],
        accuracy: f64,
    ) -> miette::Result<Proj> {
        let count = params.len();
        let params: Vec<proj_sys::PJ_PARAM_DESCRIPTION> = params
            .iter()
            .map(|p| proj_sys::PJ_PARAM_DESCRIPTION {
                name: p.name().to_cstr(),
                auth_name: p.auth_name().to_cstr(),
                code: p.code().to_cstr(),
                value: *p.value(),
                unit_name: p.unit_name().to_cstr(),
                unit_conv_factor: *p.unit_conv_factor(),
                unit_type: u32::from(*p.unit_type()),
            })
            .collect();
        let ptr = unsafe {
            proj_sys::proj_create_transformation(
                self.ptr,
                name.to_cstr(),
                auth_name.to_cstr(),
                code.to_cstr(),
                source_crs.map_or(ptr::null(), |crs| crs.ptr()),
                target_crs.map_or(ptr::null(), |crs| crs.ptr()),
                interpolation_crs.map_or(ptr::null(), |crs| crs.ptr()),
                method_name.to_cstr(),
                method_auth_name.to_cstr(),
                method_code.to_cstr(),
                count as i32,
                params.as_ptr(),
                accuracy,
            )
        };
        crate::Proj::new(self, ptr)
    }

    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_projected_crs>
    pub fn create_projected_crs(
        &self,
        crs_name: Option<&str>,
        geodetic_crs: &Proj,
        conversion: &Proj,
        coordinate_system: &Proj,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_projected_crs(
                self.ptr,
                crs_name.to_cstr(),
                geodetic_crs.ptr(),
                conversion.ptr(),
                coordinate_system.ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_bound_crs>
    pub fn crs_create_bound_crs(
        &self,
        base_crs: &Proj,
        hub_crs: &Proj,
        transformation: &Proj,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_create_bound_crs(
                self.ptr,
                base_crs.ptr(),
                hub_crs.ptr(),
                transformation.ptr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_bound_vertical_crs>
    pub fn crs_create_bound_vertical_crs(
        &self,
        vert_crs: &Proj,
        hub_geographic_3d_crs: &Proj,
        grid_name: &str,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_create_bound_vertical_crs(
                self.ptr,
                vert_crs.ptr(),
                hub_geographic_3d_crs.ptr(),
                grid_name.to_cstr(),
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_utm>
    pub fn create_conversion_utm(&self, zone: u8, north: bool) -> miette::Result<Proj> {
        if !(1..=60).contains(&zone) {
            miette::bail!("UTM zone number should between 1 and 60.");
        }
        let ptr =
            unsafe { proj_sys::proj_create_conversion_utm(self.ptr, zone as i32, north as i32) };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_transverse_mercator>
    pub fn create_conversion_transverse_mercator(
        &self,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_transverse_mercator(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_gauss_schreiber_transverse_mercator>
    pub fn create_conversion_gauss_schreiber_transverse_mercator(
        &self,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_gauss_schreiber_transverse_mercator(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_transverse_mercator_south_oriented>
    pub fn create_conversion_transverse_mercator_south_oriented(
        &self,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_transverse_mercator_south_oriented(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_two_point_equidistant>
    pub fn create_conversion_two_point_equidistant(
        &self,
        latitude_first_point: f64,
        longitude_first_point: f64,
        latitude_second_point: f64,
        longitude_second_point: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_two_point_equidistant(
                self.ptr,
                latitude_first_point,
                longitude_first_point,
                latitude_second_point,
                longitude_second_point,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_tunisia_mapping_grid>
    pub fn create_conversion_tunisia_mapping_grid(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_tunisia_mapping_grid(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_tunisia_mining_grid>
    pub fn create_conversion_tunisia_mining_grid(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_tunisia_mining_grid(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_albers_equal_area>
    pub fn create_conversion_albers_equal_area(
        &self,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_albers_equal_area(
                self.ptr,
                latitude_false_origin,
                longitude_false_origin,
                latitude_first_parallel,
                latitude_second_parallel,
                easting_false_origin,
                northing_false_origin,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_1sp>
    pub fn create_conversion_lambert_conic_conformal_1sp(
        &self,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_1sp(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_1sp_variant_b>
    pub fn create_conversion_lambert_conic_conformal_1sp_variant_b(
        &self,
        latitude_nat_origin: f64,
        scale: f64,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_1sp_variant_b(
                self.ptr,
                latitude_nat_origin,
                scale,
                latitude_false_origin,
                longitude_false_origin,
                easting_false_origin,
                northing_false_origin,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_2sp>
    pub fn create_conversion_lambert_conic_conformal_2sp(
        &self,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_2sp(
                self.ptr,
                latitude_false_origin,
                longitude_false_origin,
                latitude_first_parallel,
                latitude_second_parallel,
                easting_false_origin,
                northing_false_origin,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_2sp_michigan>
    pub fn create_conversion_lambert_conic_conformal_2sp_michigan(
        &self,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ellipsoid_scaling_factor: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_2sp_michigan(
                self.ptr,
                latitude_false_origin,
                longitude_false_origin,
                latitude_first_parallel,
                latitude_second_parallel,
                easting_false_origin,
                northing_false_origin,
                ellipsoid_scaling_factor,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_2sp_belgium>
    pub fn create_conversion_lambert_conic_conformal_2sp_belgium(
        &self,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_2sp_belgium(
                self.ptr,
                latitude_false_origin,
                longitude_false_origin,
                latitude_first_parallel,
                latitude_second_parallel,
                easting_false_origin,
                northing_false_origin,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_azimuthal_equidistant>
    pub fn create_conversion_azimuthal_equidistant(
        &self,
        latitude_nat_origin: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_azimuthal_equidistant(
                self.ptr,
                latitude_nat_origin,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_guam_projection>
    pub fn create_conversion_guam_projection(
        &self,
        latitude_nat_origin: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_guam_projection(
                self.ptr,
                latitude_nat_origin,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_bonne>
    pub fn create_conversion_bonne(
        &self,
        latitude_nat_origin: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_bonne(
                self.ptr,
                latitude_nat_origin,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_cylindrical_equal_area_spherical>
    pub fn create_conversion_lambert_cylindrical_equal_area_spherical(
        &self,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_cylindrical_equal_area_spherical(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_cylindrical_equal_area>
    pub fn create_conversion_lambert_cylindrical_equal_area(
        &self,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_cylindrical_equal_area(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_cassini_soldner>
    pub fn create_conversion_cassini_soldner(
        &self,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_cassini_soldner(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_equidistant_conic>
    pub fn create_conversion_equidistant_conic(
        &self,
        center_lat: f64,
        center_long: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_equidistant_conic(
                self.ptr,
                center_lat,
                center_long,
                latitude_first_parallel,
                latitude_second_parallel,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_i>
    pub fn create_conversion_eckert_i(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_i(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_ii>
    pub fn create_conversion_eckert_ii(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_ii(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_iii>
    pub fn create_conversion_eckert_iii(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_iii(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_iv>
    pub fn create_conversion_eckert_iv(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_iv(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_v>
    pub fn create_conversion_eckert_v(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_v(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_vi>
    pub fn create_conversion_eckert_vi(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_vi(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_equidistant_cylindrical>
    pub fn create_conversion_equidistant_cylindrical(
        &self,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_equidistant_cylindrical(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_equidistant_cylindrical_spherical>
    pub fn create_conversion_equidistant_cylindrical_spherical(
        &self,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_equidistant_cylindrical_spherical(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_gall>
    pub fn create_conversion_gall(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_gall(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_goode_homolosine>
    pub fn create_conversion_goode_homolosine(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_goode_homolosine(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_interrupted_goode_homolosine>
    pub fn create_conversion_interrupted_goode_homolosine(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_interrupted_goode_homolosine(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_geostationary_satellite_sweep_x>
    pub fn create_conversion_geostationary_satellite_sweep_x(
        &self,
        center_long: f64,
        height: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_geostationary_satellite_sweep_x(
                self.ptr,
                center_long,
                height,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_geostationary_satellite_sweep_y>
    pub fn create_conversion_geostationary_satellite_sweep_y(
        &self,
        center_long: f64,
        height: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_geostationary_satellite_sweep_y(
                self.ptr,
                center_long,
                height,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_gnomonic>
    pub fn create_conversion_gnomonic(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_gnomonic(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_hotine_oblique_mercator_variant_a>
    pub fn create_conversion_hotine_oblique_mercator_variant_a(
        &self,
        latitude_projection_centre: f64,
        longitude_projection_centre: f64,
        azimuth_initial_line: f64,
        angle_from_rectified_to_skrew_grid: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_hotine_oblique_mercator_variant_a(
                self.ptr,
                latitude_projection_centre,
                longitude_projection_centre,
                azimuth_initial_line,
                angle_from_rectified_to_skrew_grid,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_hotine_oblique_mercator_variant_b>
    pub fn create_conversion_hotine_oblique_mercator_variant_b(
        &self,
        latitude_projection_centre: f64,
        longitude_projection_centre: f64,
        azimuth_initial_line: f64,
        angle_from_rectified_to_skrew_grid: f64,
        scale: f64,
        easting_projection_centre: f64,
        northing_projection_centre: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_hotine_oblique_mercator_variant_b(
                self.ptr,
                latitude_projection_centre,
                longitude_projection_centre,
                azimuth_initial_line,
                angle_from_rectified_to_skrew_grid,
                scale,
                easting_projection_centre,
                northing_projection_centre,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_hotine_oblique_mercator_two_point_natural_origin>
    pub fn create_conversion_hotine_oblique_mercator_two_point_natural_origin(
        &self,
        latitude_projection_centre: f64,
        latitude_point1: f64,
        longitude_point1: f64,
        latitude_point2: f64,
        longitude_point2: f64,
        scale: f64,
        easting_projection_centre: f64,
        northing_projection_centre: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_hotine_oblique_mercator_two_point_natural_origin(
                self.ptr,
                latitude_projection_centre,
                latitude_point1,
                longitude_point1,
                latitude_point2,
                longitude_point2,
                scale,
                easting_projection_centre,
                northing_projection_centre,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_laborde_oblique_mercator>
    pub fn create_conversion_laborde_oblique_mercator(
        &self,
        latitude_projection_centre: f64,
        longitude_projection_centre: f64,
        azimuth_initial_line: f64,
        scale: f64,
        easting_projection_centre: f64,
        northing_projection_centre: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_laborde_oblique_mercator(
                self.ptr,
                latitude_projection_centre,
                longitude_projection_centre,
                azimuth_initial_line,
                scale,
                easting_projection_centre,
                northing_projection_centre,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_international_map_world_polyconic>
    pub fn create_conversion_international_map_world_polyconic(
        &self,
        center_long: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_international_map_world_polyconic(
                self.ptr,
                center_long,
                latitude_first_parallel,
                latitude_second_parallel,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_krovak_north_oriented>
    pub fn create_conversion_krovak_north_oriented(
        &self,
        latitude_projection_centre: f64,
        longitude_of_origin: f64,
        colatitude_cone_axis: f64,
        latitude_pseudo_standard_parallel: f64,
        scale_factor_pseudo_standard_parallel: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_krovak_north_oriented(
                self.ptr,
                latitude_projection_centre,
                longitude_of_origin,
                colatitude_cone_axis,
                latitude_pseudo_standard_parallel,
                scale_factor_pseudo_standard_parallel,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_krovak>
    pub fn create_conversion_krovak(
        &self,
        latitude_projection_centre: f64,
        longitude_of_origin: f64,
        colatitude_cone_axis: f64,
        latitude_pseudo_standard_parallel: f64,
        scale_factor_pseudo_standard_parallel: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_krovak(
                self.ptr,
                latitude_projection_centre,
                longitude_of_origin,
                colatitude_cone_axis,
                latitude_pseudo_standard_parallel,
                scale_factor_pseudo_standard_parallel,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_azimuthal_equal_area>
    pub fn create_conversion_lambert_azimuthal_equal_area(
        &self,
        latitude_nat_origin: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_azimuthal_equal_area(
                self.ptr,
                latitude_nat_origin,
                longitude_nat_origin,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_miller_cylindrical>
    pub fn create_conversion_miller_cylindrical(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_miller_cylindrical(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_mercator_variant_a>
    pub fn create_conversion_mercator_variant_a(
        &self,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_mercator_variant_a(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_mercator_variant_b>
    pub fn create_conversion_mercator_variant_b(
        &self,
        latitude_first_parallel: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_mercator_variant_b(
                self.ptr,
                latitude_first_parallel,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_popular_visualisation_pseudo_mercator>
    pub fn create_conversion_popular_visualisation_pseudo_mercator(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_popular_visualisation_pseudo_mercator(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_mollweide>
    pub fn create_conversion_mollweide(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_mollweide(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_new_zealand_mapping_grid>
    pub fn create_conversion_new_zealand_mapping_grid(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_new_zealand_mapping_grid(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_oblique_stereographic>
    pub fn create_conversion_oblique_stereographic(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_new_zealand_mapping_grid(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_orthographic>
    pub fn create_conversion_orthographic(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_orthographic(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_local_orthographic>
    pub fn create_conversion_local_orthographic(
        &self,
        center_lat: f64,
        center_long: f64,
        azimuth: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_local_orthographic(
                self.ptr,
                center_lat,
                center_long,
                azimuth,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_american_polyconic>
    pub fn create_conversion_american_polyconic(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_american_polyconic(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_polar_stereographic_variant_a>
    pub fn create_conversion_polar_stereographic_variant_a(
        &self,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_polar_stereographic_variant_a(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_polar_stereographic_variant_b>
    pub fn create_conversion_polar_stereographic_variant_b(
        &self,
        latitude_standard_parallel: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_mercator_variant_b(
                self.ptr,
                latitude_standard_parallel,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_robinson>
    pub fn create_conversion_robinson(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_robinson(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_sinusoidal>
    pub fn create_conversion_sinusoidal(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_sinusoidal(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_stereographic>
    pub fn create_conversion_stereographic(
        &self,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_stereographic(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_van_der_grinten>
    pub fn create_conversion_van_der_grinten(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_van_der_grinten(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_i>
    pub fn create_conversion_wagner_i(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_i(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_ii>
    pub fn create_conversion_wagner_ii(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_ii(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_iii>
    pub fn create_conversion_wagner_iii(
        &self,
        latitude_true_scale: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_iii(
                self.ptr,
                latitude_true_scale,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_iv>
    pub fn create_conversion_wagner_iv(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_iv(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_v>
    pub fn create_conversion_wagner_v(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_v(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_vi>
    pub fn create_conversion_wagner_vi(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_vi(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_vii>
    pub fn create_conversion_wagner_vii(
        &self,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_vii(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_quadrilateralized_spherical_cube>
    pub fn create_conversion_quadrilateralized_spherical_cube(
        &self,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_quadrilateralized_spherical_cube(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_spherical_cross_track_height>
    pub fn create_conversion_spherical_cross_track_height(
        &self,
        peg_point_lat: f64,
        peg_point_long: f64,
        peg_point_heading: f64,
        peg_point_height: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_spherical_cross_track_height(
                self.ptr,
                peg_point_lat,
                peg_point_long,
                peg_point_heading,
                peg_point_height,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_equal_earth>
    pub fn create_conversion_equal_earth(
        &self,

        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_equal_earth(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_vertical_perspective>
    pub fn create_conversion_vertical_perspective(
        &self,
        topo_origin_lat: f64,
        topo_origin_long: f64,
        topo_origin_height: f64,
        view_point_height: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_vertical_perspective(
                self.ptr,
                topo_origin_lat,
                topo_origin_long,
                topo_origin_height,
                view_point_height,
                false_easting,
                false_northing,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
                linear_unit_name.to_cstr(),
                linear_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_pole_rotation_grib_convention>
    pub fn create_conversion_pole_rotation_grib_convention(
        &self,
        south_pole_lat_in_unrotated_crs: f64,
        south_pole_long_in_unrotated_crs: f64,
        axis_rotation: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_pole_rotation_grib_convention(
                self.ptr,
                south_pole_lat_in_unrotated_crs,
                south_pole_long_in_unrotated_crs,
                axis_rotation,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_pole_rotation_netcdf_cf_convention>
    fn _create_conversion_pole_rotation_netcdf_cf_convention(
        &self,
        grid_north_pole_latitude: f64,
        grid_north_pole_longitude: f64,
        north_pole_grid_longitude: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_conversion_pole_rotation_netcdf_cf_convention(
                self.ptr,
                grid_north_pole_latitude,
                grid_north_pole_longitude,
                north_pole_grid_longitude,
                ang_unit_name.to_cstr(),
                ang_unit_conv_factor,
            )
        };
        crate::Proj::new(self, ptr)
    }
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
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_non_deprecated>
    pub fn get_non_deprecated(&self) -> miette::Result<Vec<Proj>> {
        let result = unsafe { proj_sys::proj_get_non_deprecated(self.ctx.ptr, self.ptr()) };
        pj_obj_list_to_vec(self.ctx, result)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_equivalent_to>
    pub fn is_equivalent_to(&self, other: &Proj, criterion: ComparisonCriterion) -> bool {
        unsafe { proj_sys::proj_is_equivalent_to(self.ptr(), other.ptr(), criterion.into()) != 0 }
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_equivalent_to_with_ctx>
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
        unsafe { proj_sys::proj_get_name(self.ptr()) }
            .to_string()
            .unwrap_or_default()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_id_auth_name>
    pub fn get_id_auth_name(&self, index: u16) -> Option<String> {
        unsafe { proj_sys::proj_get_id_auth_name(self.ptr(), index as i32) }.to_string()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_id_code>
    pub fn get_id_code(&self, index: u16) -> Option<String> {
        unsafe { proj_sys::proj_get_id_code(self.ptr(), index as i32) }.to_string()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_remarks>
    pub fn get_remarks(&self) -> String {
        unsafe { proj_sys::proj_get_remarks(self.ptr()) }
            .to_string()
            .unwrap_or_default()
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
        unsafe { proj_sys::proj_get_scope(self.ptr()) }.to_string()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_scope_ex>
    pub fn get_scope_ex(&self, domain_idx: u16) -> Option<String> {
        unsafe { proj_sys::proj_get_scope_ex(self.ptr(), domain_idx as i32) }.to_string()
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_area_of_use>
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
            area_name.to_string().unwrap(),
            west_lon_degree,
            south_lat_degree,
            east_lon_degree,
            north_lat_degree,
        )))
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_area_of_use_ex>
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
            area_name.to_string().unwrap(),
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
        let result = unsafe {
            proj_sys::proj_as_wkt(self.ctx.ptr, self.ptr(), wkt_type.into(), ptrs.as_ptr())
        }
        .to_string();
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
        let result = unsafe {
            proj_sys::proj_as_proj_string(
                self.ctx.ptr,
                self.ptr(),
                string_type.into(),
                ptrs.as_ptr(),
            )
        }
        .to_string();
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
        let result = unsafe { proj_sys::proj_as_projjson(self.ctx.ptr, self.ptr(), ptrs.as_ptr()) }
            .to_string();
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
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_identify>
    pub fn identify(&self, auth_name: &str) -> miette::Result<Vec<Proj>> {
        let mut confidence: Vec<i32> = Vec::new();
        let result = unsafe {
            proj_sys::proj_identify(
                self.ctx.ptr,
                self.ptr(),
                auth_name.to_cstr(),
                ptr::null(),
                &mut confidence.as_mut_ptr(),
            )
        };
        pj_obj_list_to_vec(self.ctx, result)
    }
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
            name.to_string().unwrap(),
            abbrev.to_string().unwrap(),
            AxisDirection::from_str(&direction.to_string().unwrap()).into_diagnostic()?,
            unit_conv_factor,
            unit_name.to_string().unwrap(),
            unit_auth_name.to_string().unwrap(),
            unit_code.to_string().unwrap(),
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
        unsafe { proj_sys::proj_get_celestial_body_name(self.ctx.ptr, self.ptr()) }.to_string()
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
            unit_name.to_string().unwrap_or_default(),
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
            method_name.to_string().unwrap_or_default(),
            method_auth_name.to_string().unwrap_or_default(),
            method_code.to_string().unwrap_or_default(),
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
            proj_sys::proj_coordoperation_get_param_index(self.ctx.ptr, self.ptr(), name.to_cstr())
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
            name.to_string().unwrap_or_default(),
            auth_name.to_string().unwrap_or_default(),
            code.to_string().unwrap_or_default(),
            value,
            (value_string).to_string().unwrap_or_default(),
            unit_conv_factor,
            (unit_name).to_string().unwrap_or_default(),
            (unit_auth_name).to_string().unwrap_or_default(),
            (unit_code).to_string().unwrap_or_default(),
            UnitCategory::from_str(&(unit_category).to_string().unwrap()).into_diagnostic()?,
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
            (short_name).to_string().unwrap_or_default(),
            (full_name).to_string().unwrap_or_default(),
            (package_name).to_string().unwrap_or_default(),
            (url).to_string().unwrap_or_default(),
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
    ///# See Also
    ///
    /// * [`Self::crs_is_derived`]
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_derived_crs>
    fn _is_derived_crs(&self) { unimplemented!("Use other function to instead.") }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_alter_name>
    pub fn alter_name(&self, name: &str) -> miette::Result<Proj> {
        let ptr = unsafe { proj_sys::proj_alter_name(self.ctx.ptr, self.ptr(), name.to_cstr()) };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_alter_id>
    pub fn alter_id(&self, auth_name: &str, code: &str) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_alter_id(
                self.ctx.ptr,
                self.ptr(),
                auth_name.to_cstr(),
                code.to_cstr(),
            )
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_alter_geodetic_crs>
    pub fn crs_alter_geodetic_crs(&self, new_geod_crs: &Proj) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_alter_geodetic_crs(self.ctx.ptr, self.ptr(), new_geod_crs.ptr())
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_alter_cs_angular_unit>
    pub fn crs_alter_cs_angular_unit(
        &self,
        angular_unit: Option<&str>,
        angular_units_convs: f64,
        unit_auth_name: Option<&str>,
        unit_code: Option<&str>,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_alter_cs_angular_unit(
                self.ctx.ptr,
                self.ptr(),
                angular_unit.to_cstr(),
                angular_units_convs,
                unit_auth_name.to_cstr(),
                unit_code.to_cstr(),
            )
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_alter_cs_linear_unit>
    pub fn crs_alter_cs_linear_unit(
        &self,
        linear_units: Option<&str>,
        linear_units_conv: f64,
        unit_auth_name: Option<&str>,
        unit_code: Option<&str>,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_alter_cs_linear_unit(
                self.ctx.ptr,
                self.ptr(),
                linear_units.to_cstr(),
                linear_units_conv,
                unit_auth_name.to_cstr(),
                unit_code.to_cstr(),
            )
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_alter_parameters_linear_unit>
    pub fn crs_alter_parameters_linear_unit(
        &self,
        linear_units: Option<&str>,
        linear_units_conv: f64,
        unit_auth_name: Option<&str>,
        unit_code: Option<&str>,
        convert_to_new_unit: bool,
    ) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_alter_parameters_linear_unit(
                self.ctx.ptr,
                self.ptr(),
                linear_units.to_cstr(),
                linear_units_conv,
                unit_auth_name.to_cstr(),
                unit_code.to_cstr(),
                convert_to_new_unit as i32,
            )
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_promote_to_3D>
    pub fn crs_promote_to_3d(&self, crs_3d_name: Option<&str>) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_promote_to_3D(self.ctx.ptr, crs_3d_name.to_cstr(), self.ptr())
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_demote_to_2D>
    pub fn crs_demote_to_2d(&self, crs_2d_name: Option<&str>) -> miette::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_demote_to_2D(self.ctx.ptr, crs_2d_name.to_cstr(), self.ptr())
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_convert_conversion_to_other_method>
    pub fn convert_conversion_to_other_method(
        &self,
        new_method_epsg_code: Option<u16>,
        new_method_name: Option<&str>,
    ) -> miette::Result<Proj> {
        if new_method_epsg_code.is_none() && new_method_name.is_none() {
            miette::bail!(
                "At least one of `new_method_epsg_code` and  `new_method_name` must be set."
            )
        }
        let ptr = unsafe {
            proj_sys::proj_convert_conversion_to_other_method(
                self.ctx.ptr,
                self.ptr(),
                new_method_epsg_code.unwrap_or_default() as i32,
                new_method_name.to_cstr(),
            )
        };
        Proj::new(self.ctx, ptr)
    }
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

        Proj::new(self.ctx, ptr).unwrap()
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
///# References
///
/// <>
fn _insert_object_session_create() { todo!() }
///# References
///
/// <>
fn _string_destroy() { todo!() }
///# References
///
/// <>
fn _operation_factory_context_destroy() { todo!() }
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

#[cfg(test)]
mod test_context_basic {
    use strum::IntoEnumIterator;

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
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::ProjDataVersion)
            .unwrap();
        assert_eq!(data, "1.21");

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
    #[test]
    fn test_uom_get_info_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let info = ctx.uom_get_info_from_database("EPSG", "9102")?;
        println!("{:?}", info);
        assert_eq!(info.name(), "degree");
        assert_eq!(info.conv_factor(), &0.017453292519943295);
        assert_eq!(info.category(), &UomCategory::Angular);
        Ok(())
    }
    #[test]
    fn test_grid_get_info_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let info = ctx.grid_get_info_from_database("au_icsm_GDA94_GDA2020_conformal.tif")?;
        println!("{:?}", info);
        assert_eq!(
            info.full_name(),
            "https://cdn.proj.org/au_icsm_GDA94_GDA2020_conformal.tif"
        );
        assert_eq!(
            info.url(),
            "https://cdn.proj.org/au_icsm_GDA94_GDA2020_conformal.tif"
        );
        assert!(info.direct_download());
        assert!(info.open_license());
        assert!(info.available());
        Ok(())
    }
    #[test]
    fn test_create_from_name() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj_list = ctx.create_from_name(None, "WGS 84", None, false, 0)?;
        println!(
            "{}",
            pj_list.first().unwrap().as_wkt(
                WktType::Wkt2_2019,
                None,
                None,
                None,
                None,
                None,
                None
            )?
        );
        assert!(!pj_list.is_empty());
        Ok(())
    }
    #[test]
    fn test_get_geoid_models_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let models = ctx.get_geoid_models_from_database("EPSG", "5703")?;
        assert_eq!(
            models,
            vec![
                "GEOID03", "GEOID06", "GEOID09", "GEOID12A", "GEOID12B", "GEOID18", "GEOID99",
                "GGM10"
            ]
        );
        Ok(())
    }
    #[test]
    fn test_get_authorities_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let authorities = ctx.get_authorities_from_database()?;
        assert_eq!(
            authorities,
            vec![
                "EPSG", "ESRI", "IAU_2015", "IGNF", "NKG", "NRCAN", "OGC", "PROJ"
            ]
        );
        Ok(())
    }
    #[test]
    fn test_get_codes_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        for t in ProjType::iter() {
            let codes = ctx.get_codes_from_database("EPSG", t.clone(), true);
            if codes.is_err() {
                println!("{:?}", t);
            } else {
                let result = codes?;
                println!("{:?}:{}", t, result.len());
                assert!(!result.is_empty());
            }
        }
        Ok(())
    }
    #[test]
    fn test_get_celestial_body_list_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let list = ctx.get_celestial_body_list_from_database("ESRI")?;
        println!("{:?}", list.first().unwrap());
        assert!(!list.is_empty());
        Ok(())
    }
    #[test]
    fn test_get_crs_info_list_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let list = ctx.get_crs_info_list_from_database(Some("EPSG"), None)?;
        println!("{:?}", list.first().unwrap());
        assert!(!list.is_empty());
        Ok(())
    }
    #[test]
    fn test_get_units_from_database() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let units = ctx.get_units_from_database("EPSG", UnitCategory::Linear, true)?;
        println!("{:?}", units.first().unwrap());
        assert!(!units.is_empty());
        Ok(())
    }
}
#[cfg(test)]
mod test_context_advanced {
    use strum::IntoEnumIterator;

    use super::*;
    #[test]
    fn test_create_cs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        for a in AxisDirection::iter() {
            let pj: Proj<'_> = ctx.create_cs(
                CoordinateSystemType::Cartesian,
                &[
                    AxisDescription::new(
                        Some("Longitude"),
                        Some("lon"),
                        a,
                        Some("Degree"),
                        1.0,
                        UnitType::Angular,
                    )?,
                    AxisDescription::new(
                        Some("Latitude"),
                        Some("lat"),
                        AxisDirection::North,
                        Some("Degree"),
                        1.0,
                        UnitType::Angular,
                    )?,
                ],
            )?;
            let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
            assert!(wkt.contains("9122"));
        }
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
        let pj = ctx.create_ellipsoidal_3d_cs(
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
    fn test_query_geodetic_crs_from_datum() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj_list =
            ctx.query_geodetic_crs_from_datum(Some("EPSG"), "EPSG", "6326", Some("geographic 2D"))?;
        println!(
            "{}",
            pj_list.first().unwrap().as_wkt(
                WktType::Wkt2_2019,
                None,
                None,
                None,
                None,
                None,
                None
            )?
        );
        assert!(!pj_list.is_empty());
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
    #[test]
    fn test_create_create_geocentric_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> = ctx.create_geocentric_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("9122"));
        Ok(())
    }
    #[test]
    fn test_create_geocentric_crs_from_datum() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj1: Proj<'_> = ctx.create_geocentric_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let pj2: Proj<'_> = ctx.create_geocentric_crs_from_datum(
            Some("WGS 84"),
            &pj1.crs_get_datum()?.unwrap(),
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj2.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("9122"));
        Ok(())
    }
    #[test]
    fn test_create_derived_geographic_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let crs_4326 = ctx.create("EPSG:4326")?;
        let conversion = ctx.create_conversion_pole_rotation_grib_convention(
            2.0,
            3.0,
            4.0,
            Some("Degree"),
            0.0174532925199433,
        )?;
        let cs = crs_4326.crs_get_coordinate_system()?;
        let pj: Proj<'_> =
            ctx.create_derived_geographic_crs(Some("my rotated CRS"), &crs_4326, &conversion, &cs)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("my rotated CRS"));
        Ok(())
    }
    #[test]
    fn test_crs_create_projected_3d_crs_from_2d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let proj_crs = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let geog_3d_crs = ctx.create_from_database("EPSG", "4979", Category::Crs, false)?;
        let pj: Proj<'_> =
            ctx.crs_create_projected_3d_crs_from_2d(None, &proj_crs, Some(&geog_3d_crs))?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("WGS 84 / UTM zone 31N"));
        Ok(())
    }
    #[test]
    fn test_create_engineering_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;

        let pj: Proj<'_> = ctx.create_engineering_crs(Some("name"))?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("Unknown engineering datum"));
        Ok(())
    }
    #[test]
    fn test_create_vertical_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> =
            ctx.create_vertical_crs(Some("myVertCRS"), Some("myVertDatum"), None, 0.0)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("myVertDatum"));
        Ok(())
    }
    #[test]
    fn test_create_vertical_crs_ex() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> = ctx.create_vertical_crs_ex(
            Some("myVertCRS (ftUS)"),
            Some("myVertDatum"),
            None,
            None,
            Some("US survey foot"),
            0.304800609601219,
            Some("PROJ @foo.gtx"),
            None,
            None,
            None,
            None,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("myVertCRS (ftUS)"));
        Ok(())
    }
    #[test]
    fn test_create_compound_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let horiz_crs = ctx.create_from_database("EPSG", "6340", Category::Crs, false)?;
        let vert_crs: Proj<'_> = ctx.create_vertical_crs_ex(
            Some("myVertCRS (ftUS)"),
            Some("myVertDatum"),
            None,
            None,
            Some("US survey foot"),
            0.304800609601219,
            Some("PROJ @foo.gtx"),
            None,
            None,
            None,
            None,
        )?;
        let pj: Proj<'_> = ctx.create_compound_crs(Some("Compound"), &horiz_crs, &vert_crs)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("Compound"));
        Ok(())
    }
    #[test]
    fn test_create_conversion() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj<'_> = ctx.create_conversion(
            Some("conv"),
            Some("conv auth"),
            Some("conv code"),
            Some("method"),
            Some("method auth"),
            Some("method code"),
            &[ParamDescription::new(
                Some("param name".to_string()),
                None,
                None,
                0.99,
                None,
                1.0,
                UnitType::Scale,
            )],
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("conv"));
        Ok(())
    }
    #[test]
    fn test_create_transformation() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let geog_cs =
            ctx.create_ellipsoidal_2d_cs(EllipsoidalCs2dType::LongitudeLatitude, None, 0.0)?;
        let source_crs = ctx.create_geographic_crs(
            Some("Source CRS"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("Degree"),
            0.0174532925199433,
            &geog_cs,
        )?;
        let target_crs = ctx.create_geographic_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("Degree"),
            0.0174532925199433,
            &geog_cs,
        )?;
        let pj = ctx.create_transformation(
            Some("transf"),
            Some("transf auth"),
            Some("conv code"),
            Some(&source_crs),
            Some(&target_crs),
            Some(&target_crs),
            Some("method"),
            Some("method auth"),
            Some("method code"),
            &[ParamDescription::new(
                Some("param name".to_string()),
                None,
                None,
                0.99,
                None,
                1.0,
                UnitType::Scale,
            )],
            0.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("transf"));
        Ok(())
    }
    #[test]
    fn test_projected_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let conv = ctx.create_conversion(
            Some("conv"),
            Some("conv auth"),
            Some("conv code"),
            Some("method"),
            Some("method auth"),
            Some("method code"),
            &[ParamDescription::new(
                Some("param name".to_string()),
                None,
                None,
                0.99,
                None,
                1.0,
                UnitType::Scale,
            )],
        )?;
        let geog_cs =
            ctx.create_ellipsoidal_2d_cs(EllipsoidalCs2dType::LongitudeLatitude, None, 0.0)?;

        let geog_crs = ctx.create_geographic_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("Degree"),
            0.0174532925199433,
            &geog_cs,
        )?;
        let cs = ctx.create_cartesian_2d_cs(CartesianCs2dType::EastingNorthing, None, 0.0)?;
        let pj: Proj<'_> = ctx.create_projected_crs(Some("my CRS"), &geog_crs, &conv, &cs)?;

        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("my CRS"));
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
    fn test_get_non_deprecated() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4226")?;
        let pj_list = pj.get_non_deprecated()?;
        println!(
            "{}",
            pj_list.first().unwrap().as_wkt(
                WktType::Wkt2_2019,
                None,
                None,
                None,
                None,
                None,
                None
            )?
        );
        assert!(!pj_list.is_empty());
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
    fn test_get_area_of_use() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let area = pj.get_area_of_use()?.unwrap();

        assert_eq!(area.area_name(), "World.");
        assert_eq!(area.east_lon_degree(), &180.0);
        assert_eq!(area.west_lon_degree(), &-180.0);
        assert_eq!(area.north_lat_degree(), &90.0);
        assert_eq!(area.south_lat_degree(), &-90.0);
        Ok(())
    }
    #[test]
    fn test_get_area_of_use_ex() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "6316", Category::Crs, false)?;
        let area = pj.get_area_of_use_ex(1)?.unwrap();

        assert_eq!(area.area_name(), "North Macedonia.");
        assert_eq!(area.east_lon_degree(), &23.04);
        assert_eq!(area.west_lon_degree(), &20.45);
        assert_eq!(area.north_lat_degree(), &42.36);
        assert_eq!(area.south_lat_degree(), &40.85);
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
    fn test_identify() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let pj_list = pj.identify("EPSG")?;
        println!(
            "{}",
            pj_list.first().unwrap().as_wkt(
                WktType::Wkt2_2019,
                None,
                None,
                None,
                None,
                None,
                None
            )?
        );
        assert!(!pj_list.is_empty());
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
    fn test_crs_get_sub_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let horiz_crs = ctx.create_from_database("EPSG", "6340", Category::Crs, false)?;
        let vert_crs: Proj<'_> = ctx.create_vertical_crs_ex(
            Some("myVertCRS (ftUS)"),
            Some("myVertDatum"),
            None,
            None,
            Some("US survey foot"),
            0.304800609601219,
            Some("PROJ @foo.gtx"),
            None,
            None,
            None,
            None,
        )?;
        let compound: Proj<'_> =
            ctx.create_compound_crs(Some("Compound"), &horiz_crs, &vert_crs)?;
        let pj = compound.crs_get_sub_crs(0)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("NAD83"));
        Ok(())
    }
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
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let op = pj.crs_get_coordoperation()?;
        let param = op.coordoperation_get_param(1)?;
        assert_eq!(param.name(), "Longitude of natural origin");
        assert_eq!(param.code(), "8802");

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
        // let pj = ctx.create_from_database("EPSG", "8048",
        // Category::CoordinateOperation, false)?; let step =
        // pj.concatoperation_get_step(1)?; let wkt = step.
        // as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
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
    use strum::IntoEnumIterator;

    use super::*;
    #[test]
    fn test_alter_name() -> miette::Result<()> {
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
        let pj = pj.alter_name("new name")?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}",);
        assert!(wkt.contains("new name"));
        Ok(())
    }
    #[test]
    fn test_alter_id() -> miette::Result<()> {
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
        let pj = pj.alter_id("new_auth", "new_code")?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}",);
        assert!(wkt.contains("new_auth"));
        Ok(())
    }
    #[test]
    fn test_crs_alter_geodetic_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let new_geod_crs = ctx.create("+proj=latlong +type=crs")?;
        let pj_alterd = pj.crs_alter_geodetic_crs(&new_geod_crs)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        let wkt_alterd =
            pj_alterd.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}",);
        println!("{wkt_alterd}",);
        assert_ne!(wkt.to_string(), wkt_alterd.to_string());
        Ok(())
    }
    #[test]
    fn test_crs_alter_cs_angular_unit() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let pj_alterd =
            pj.crs_alter_cs_angular_unit(Some("my unit"), 2.0, Some("my auth"), Some("my code"))?;
        let wkt = pj_alterd.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("my unit"));
        Ok(())
    }
    #[test]
    fn test_crs_alter_cs_linear_unit() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let pj_alterd =
            pj.crs_alter_cs_linear_unit(Some("my unit"), 2.0, Some("my auth"), Some("my code"))?;
        let wkt = pj_alterd.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("my unit"));

        Ok(())
    }
    #[test]
    fn test_crs_alter_parameters_linear_unit() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        let pj_alterd = pj.crs_alter_parameters_linear_unit(
            Some("my unit"),
            2.0,
            Some("my auth"),
            Some("my code"),
            false,
        )?;
        let wkt = pj_alterd.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("my unit"));

        Ok(())
    }
    #[test]
    fn test_crs_promote_to_3d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let pj_3d = pj.crs_promote_to_3d(None)?;
        let wkt = pj_3d.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("CS[ellipsoidal,3]"));
        Ok(())
    }
    #[test]
    fn test_crs_demote_to_2d() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4979")?;
        let pj_2d = pj.crs_demote_to_2d(None)?;
        let wkt = pj_2d.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("CS[ellipsoidal,2]"));
        Ok(())
    }
    #[test]
    fn test_convert_conversion_to_other_method() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        {
            let conv = ctx.create_conversion_mercator_variant_a(
                0.0,
                1.0,
                0.99,
                2.0,
                3.0,
                Some("Degree"),
                0.0174532925199433,
                Some("Metre"),
                1.0,
            )?;
            let geog_cs =
                ctx.create_ellipsoidal_2d_cs(EllipsoidalCs2dType::LongitudeLatitude, None, 0.0)?;

            let geog_crs = ctx.create_geographic_crs(
                Some("WGS 84"),
                Some("World Geodetic System 1984"),
                Some("WGS 84"),
                6378137.0,
                298.257223563,
                Some("Greenwich"),
                0.0,
                Some("Degree"),
                0.0174532925199433,
                &geog_cs,
            )?;
            let cs = ctx.create_cartesian_2d_cs(CartesianCs2dType::EastingNorthing, None, 0.0)?;
            let pj: Proj<'_> = ctx.create_projected_crs(Some("my CRS"), &geog_crs, &conv, &cs)?;
            let conv_in_proj = pj.crs_get_coordoperation()?;
            //by code
            {
                let new_conv = conv_in_proj.convert_conversion_to_other_method(Some(9805), None)?;
                let wkt =
                    new_conv.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
                println!("{wkt}");
                assert!(wkt.contains("9805"));
            }
            //both none
            {
                let new_conv = conv_in_proj.convert_conversion_to_other_method(None, None);
                assert!(new_conv.is_err());
            }
        }
        Ok(())
    }

    #[test]
    fn test_crs_create_bound_crs_to_wgs84() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_from_database("EPSG", "32631", Category::Crs, false)?;
        for a in AllowIntermediateCrs::iter() {
            let bound = pj.crs_create_bound_crs_to_wgs84(Some(a))?;
            let wkt = bound.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
            println!("{wkt}");
            assert!(wkt.contains("32631"));
        }

        Ok(())
    }
}
