use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str::FromStr;

use envoy::{AsVecPtr, CStrListToVecString, CStrToString, ToCString};
use miette::IntoDiagnostic;

use crate::data_types::iso19111::*;
use crate::{OwnedCStrings, Proj, ProjOptions, pj_obj_list_to_vec};
/// # ISO-19111 Base functions
impl crate::Context {
    ///Explicitly point to the main PROJ CRS and coordinate operation
    /// definition database ("proj.db"), and potentially auxiliary databases
    /// with same structure.
    ///
    ///Starting with PROJ 8.1, if the auxDbPaths parameter is an empty array,
    /// the PROJ_AUX_DB environment variable will be used, if set. It must
    /// contain one or several paths. If several paths are provided, they must
    /// be separated by the colon (:) character on Unix, and on Windows, by the
    /// semi-colon (;) character.
    ///
    /// # Arguments
    ///
    /// * `db_path`: Path to main database.
    /// * `aux_db_paths`: List of auxiliary database filenames, or `None`.
    ///
    /// # References
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
                .map(|f| f.to_str().unwrap().to_cstring())
                .collect()
        });

        let aux_db_paths_ptr: Option<Vec<*const i8>> =
            aux_db_paths.map(|aux_db_paths| aux_db_paths.iter().map(|f| f.as_ptr()).collect());

        let result = unsafe {
            proj_sys::proj_context_set_database_path(
                self.ptr,
                db_path.to_str().unwrap().to_cstring().as_ptr(),
                aux_db_paths_ptr.map_or(ptr::null(), |ptr| ptr.as_ptr()),
                ptr::null(),
            )
        };
        if result != 1 {
            miette::bail!("Error");
        }
        Ok(self)
    }
    ///Returns the path to the database.
    ///
    ///The returned pointer remains valid while ctx is valid, and until
    /// [`Self::set_database_path()`] is called.
    ///
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
    ///Return a metadata from the database.
    ///
    /// # Arguments
    ///
    /// * `key`: Metadata key. Must not be NULL
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_database_metadata>
    pub fn get_database_metadata(&self, key: DatabaseMetadataKey) -> Option<String> {
        unsafe {
            proj_sys::proj_context_get_database_metadata(
                self.ptr,
                key.as_ref().to_cstring().as_ptr(),
            )
        }
        .to_string()
    }
    ///Return the database structure.
    ///
    ///Return SQL statements to run to initiate a new valid auxiliary empty
    /// database. It contains definitions of tables, views and triggers, as well
    /// as metadata for the version of the layout of the database.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_database_structure>
    pub fn get_database_structure(&self) -> miette::Result<Vec<String>> {
        let ptr = unsafe { proj_sys::proj_context_get_database_structure(self.ptr, ptr::null()) };
        let out_vec = ptr.to_vec_string();
        unsafe {
            proj_sys::proj_string_list_destroy(ptr);
        }
        Ok(out_vec)
    }
    ///Guess the "dialect" of the WKT string.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_guess_wkt_dialect>
    pub fn guess_wkt_dialect(&self, wkt: &str) -> miette::Result<GuessedWktDialect> {
        GuessedWktDialect::try_from(unsafe {
            proj_sys::proj_context_guess_wkt_dialect(self.ptr, wkt.to_cstring().as_ptr())
        })
        .into_diagnostic()
    }
    ///Instantiate an object from a WKT string.
    ///
    /// The returned object must be unreferenced with proj_destroy() after use.
    /// It should be used by at most one thread at a time.
    ///
    ///The distinction between warnings and grammar errors is somewhat
    /// artificial and does not tell much about the real criticity of the
    /// non-compliance. Some warnings may be more concerning than some grammar
    /// errors. Human expertise (or, by the time this comment will be read,
    /// specialized AI) is generally needed to perform that assessment.
    ///
    /// # Arguments
    ///
    /// * `wkt`: WKT string
    /// * `strict/: Defaults to `false`. When set to `true`, strict validation
    ///   will be enabled.
    /// * `unset_identifiers_if_incompatible_def`: Defaults to `true`. When set
    ///   to `true`, object identifiers are unset when there is a contradiction
    ///   between the definition from WKT and the one from the database.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_from_wkt>
    pub fn create_from_wkt(
        &self,
        wkt: &str,
        strict: Option<bool>,
        unset_identifiers_if_incompatible_def: Option<bool>,
    ) -> miette::Result<Proj<'_>> {
        let mut options = ProjOptions::new(2);
        options.push_optional_pass(strict, "STRICT");
        options.push_optional_pass(
            unset_identifiers_if_incompatible_def,
            "UNSET_IDENTIFIERS_IF_INCOMPATIBLE_DEF",
        );
        let mut out_warnings: *mut *mut i8 = std::ptr::null_mut();
        let mut out_grammar_errors: *mut *mut i8 = std::ptr::null_mut();
        let ptr = unsafe {
            proj_sys::proj_create_from_wkt(
                self.ptr,
                wkt.to_cstring().as_ptr(),
                options.as_vec_ptr().as_ptr(),
                &mut out_warnings,
                &mut out_grammar_errors,
            )
        };
        out_warnings
            .to_vec_string()
            .iter()
            .for_each(|w| clerk::warn!("{w}"));

        out_grammar_errors
            .to_vec_string()
            .iter()
            .for_each(|w| clerk::error!("{w}"));

        Proj::new(self, ptr)
    }
    ///Instantiate an object from a database lookup.
    ///
    /// The returned object must be unreferenced with proj_destroy() after use.
    /// It should be used by at most one thread at a time.
    ///
    /// # Arguments
    ///
    /// * `auth_name`: Authority name
    /// * `code`: Object code
    /// * `category`: Object category
    /// * `use_projalternative_grid_names`: Whether PROJ alternative grid names
    ///   should be substituted to the official grid names. Only used on
    ///   transformations
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_from_database>
    pub fn create_from_database(
        &self,
        auth_name: &str,
        code: &str,
        category: Category,
        use_projalternative_grid_names: bool,
    ) -> miette::Result<Proj<'_>> {
        let ptr = unsafe {
            proj_sys::proj_create_from_database(
                self.ptr,
                auth_name.to_cstring().as_ptr(),
                code.to_cstring().as_ptr(),
                category.into(),
                use_projalternative_grid_names as i32,
                ptr::null(),
            )
        };
        Proj::new(self, ptr)
    }
    ///Get information for a unit of measure from a database lookup.
    ///
    /// # Arguments
    ///
    /// * `auth_name`: Authority name
    /// * `code`: Unit of measure code
    ///
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
                auth_name.to_cstring().as_ptr(),
                code.to_cstring().as_ptr(),
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
    ///Get information for a grid from a database lookup.
    ///
    /// # Arguments
    ///
    /// * `grid_name`: Grid name
    ///
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
                grid_name.to_cstring().as_ptr(),
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
    ///Return a list of objects by their name.
    ///
    /// # Arguments
    /// * `auth_name`: Authority name, used to restrict the search. Or `None`
    ///   for all authorities.
    /// * `searched_name`: Searched name. Must be at least 2 character long.
    /// * `types`: List of object types into which to search. If `None`, all
    ///   object types will be searched.
    /// * `approximate_match`: Whether approximate name identification is
    ///   allowed.
    /// * `limit_result_count`: Maximum number of results to return. Or 0 for
    ///   unlimited.
    ///
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
    ) -> miette::Result<Vec<Proj<'_>>> {
        let (types, count) = types.map_or((None, 0), |types| {
            let types: Vec<u32> = types.iter().map(|f| u32::from(f.clone())).collect();
            let count = types.len();
            (Some(types), count)
        });
        let auth_name = auth_name.map(|s| s.to_cstring());
        let result = unsafe {
            proj_sys::proj_create_from_name(
                self.ptr,
                auth_name.map_or(ptr::null(), |s| s.as_ptr()),
                searched_name.to_cstring().as_ptr(),
                types.map_or(ptr::null(), |types| types.as_ptr()),
                count,
                approximate_match as i32,
                limit_result_count,
                ptr::null(),
            )
        };
        pj_obj_list_to_vec(self, result)
    }
    ///Returns a list of geoid models available for that crs.
    ///
    ///The list includes the geoid models connected directly with the crs, or
    /// via "Height Depth Reversal" or "Change of Vertical Unit"
    /// transformations.
    ///
    /// # Arguments
    ///
    /// * auth_name: Authority name
    /// * code: Object code
    ///
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
                auth_name.to_cstring().as_ptr(),
                code.to_cstring().as_ptr(),
                ptr::null(),
            )
        };
        if ptr.is_null() {
            miette::bail!("Error");
        }
        let out_vec = ptr.to_vec_string();
        unsafe {
            proj_sys::proj_string_list_destroy(ptr);
        }
        Ok(out_vec)
    }
    ///  Return the list of authorities used in the database.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_authorities_from_database>
    pub fn get_authorities_from_database(&self) -> miette::Result<Vec<String>> {
        let ptr = unsafe { proj_sys::proj_get_authorities_from_database(self.ptr) };
        if ptr.is_null() {
            miette::bail!("Error");
        }
        let out_vec = ptr.to_vec_string();
        unsafe {
            proj_sys::proj_string_list_destroy(ptr);
        }
        Ok(out_vec)
    }
    /// Returns the set of authority codes of the given object type.
    ///
    /// # Arguments
    ///
    /// * `auth_name`: Authority name
    /// * `type`: Object type.
    /// * `allow_deprecated`: whether we should return deprecated objects as
    ///   well.
    ///
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
                auth_name.to_cstring().as_ptr(),
                proj_type.into(),
                allow_deprecated as i32,
            )
        };
        if ptr.is_null() {
            miette::bail!("Error");
        }
        let out_vec = ptr.to_vec_string();
        unsafe {
            proj_sys::proj_string_list_destroy(ptr);
        }
        Ok(out_vec)
    }
    ///Enumerate celestial bodies from the database.
    ///
    /// # Arguments
    ///
    /// * `auth_name`: Authority name, used to restrict the search. Or `None`
    ///   for all authorities.
    /// * `out_result_count`: Output parameter pointing to an integer to receive
    ///   the size of the result list. Might be `None`
    ///
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
                auth_name.to_cstring().as_ptr(),
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
    ///Enumerate CRS objects from the database, taking into account various
    /// criteria.
    ///
    /// When no filter parameters are set, this is functionally equivalent to
    /// proj_get_codes_from_database(), instantiating a PJ* object for each of
    /// the codes with proj_create_from_database() and retrieving information
    /// with the various getters. However this function will be much faster.
    ///
    /// # Arguments
    ///
    /// * auth_name: Authority name, used to restrict the search. Or `None` for
    ///   all authorities.
    /// * params: Additional criteria, or `None`. If not-None, params SHOULD have
    ///   been allocated by proj_get_crs_list_parameters_create(), as the
    ///   PROJ_CRS_LIST_PARAMETERS structure might grow over time.
    ///
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
        let mut owned = OwnedCStrings::new();
        let ptr = unsafe {
            proj_sys::proj_get_crs_info_list_from_database(
                self.ptr,
                owned.push_option(auth_name),
                params.map_or(ptr::null(), |p| {
                    let types: Vec<u32> = p
                        .types()
                        .to_owned()
                        .iter()
                        .map(|f| u32::from(f.clone()))
                        .collect();

                    &proj_sys::PROJ_CRS_LIST_PARAMETERS {
                        types: types.as_ptr(),
                        typesCount: p.types().len(),
                        crs_area_of_use_contains_bbox: *p.west_lon_degree() as i32,
                        bbox_valid: *p.bbox_valid() as i32,
                        west_lon_degree: *p.west_lon_degree(),
                        south_lat_degree: *p.south_lat_degree(),
                        east_lon_degree: *p.east_lon_degree(),
                        north_lat_degree: *p.north_lat_degree(),
                        allow_deprecated: *p.allow_deprecated() as i32,
                        celestial_body_name: p
                            .celestial_body_name()
                            .clone()
                            .map_or(ptr::null(), |s| s.as_ptr()),
                    }
                }),
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
    ///Enumerate units from the database, taking into account various criteria.
    ///
    /// # Arguments
    ///
    /// * `auth_name`: Authority name, used to restrict the search. Or `None`
    ///   for all authorities.
    /// * `category`: Filter by category, if this parameter is not `None`.
    /// * `allow_deprecated`: whether we should return deprecated objects as
    ///   well.
    ///
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
                auth_name.to_cstring().as_ptr(),
                category.as_ref().to_cstring().as_ptr(),
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

    ///Return an object from the result set.
    ///
    /// # Arguments
    ///
    /// * `result`:  Object of type PJ_OBJ_LIST
    /// * `index`:  Index
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_get>
    pub(crate) fn list_get(
        &self,
        result: *const proj_sys::PJ_OBJ_LIST,
        index: i32,
    ) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_list_get(self.ptr, result, index) };
        Proj::new(self, ptr)
    }
    ///# References
    ///
    /// <>
    fn _get_suggested_operation(&self) { todo!() }
}

#[cfg(test)]
mod test {
    use proj_sys::{PROJ_VERSION_MAJOR, PROJ_VERSION_MINOR, PROJ_VERSION_PATCH};
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
            .get_database_metadata(DatabaseMetadataKey::ProjVersion)
            .unwrap();
        assert_eq!(
            data,
            format!(
                "{}.{}.{}",
                PROJ_VERSION_MAJOR, PROJ_VERSION_MINOR, PROJ_VERSION_PATCH
            )
        );
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
        //invalid
        assert!(ctx.create_from_wkt("invalid wkt", None, None).is_err());
        //valid
        ctx.create_from_wkt(
            &ctx.create("EPSG:4326")?.as_wkt(
                WktType::Wkt2_2019,
                None,
                None,
                None,
                None,
                None,
                None,
            )?,
            None,
            None,
        )?;
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
