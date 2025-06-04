use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str::FromStr;

use envoy::{CStrListToVecString, CStrToString, ToCStr};
use miette::IntoDiagnostic;

use super::string_list_destroy;
use crate::data_types::iso19111::*;
use crate::{Proj, ProjOptions, pj_obj_list_to_vec};
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
                ptr::null(),
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
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_suggests_code_for>
    pub fn suggests_code_for(&self, object: &Proj, authority: &str, numeric_code: bool) -> String {
        let result = unsafe {
            proj_sys::proj_suggests_code_for(
                self.ptr,
                object.ptr(),
                authority.to_cstr(),
                numeric_code as i32,
                ptr::null(),
            )
        };
        result.to_string().expect("Error")
    }
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

#[cfg(test)]
mod test {
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
        assert_eq!(data, "v12.012");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::EpsgDate)
            .unwrap();
        assert_eq!(data, "2025-05-21");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::EsriVersion)
            .unwrap();
        assert_eq!(data, "ArcGIS Pro 3.5");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::EsriDate)
            .unwrap();
        assert_eq!(data, "2025-05-11");
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
        assert_eq!(data, "9.6.1");
        let data = ctx
            .get_database_metadata(DatabaseMetadataKey::ProjDataVersion)
            .unwrap();
        assert_eq!(data, "1.22");

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
    fn test_suggests_code_for() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let wkt = "GEOGCRS[\"myGDA2020\",
                       DATUM[\"GDA2020\",
                           ELLIPSOID[\"GRS_1980\",6378137,298.257222101,
                               LENGTHUNIT[\"metre\",1]]],
                       PRIMEM[\"Greenwich\",0,
                           ANGLEUNIT[\"Degree\",0.0174532925199433]],
                       CS[ellipsoidal,2],
                           AXIS[\"geodetic latitude (Lat)\",north,
                               ORDER[1],
                               ANGLEUNIT[\"degree\",0.0174532925199433]],
                           AXIS[\"geodetic longitude (Lon)\",east,
                               ORDER[2],
                               ANGLEUNIT[\"degree\",0.0174532925199433]]]";
        println!("{wkt}");
        let crs = ctx.create_from_wkt(wkt, None, None)?;
        let code = ctx.suggests_code_for(&crs, "HOBU", true);
        assert_eq!(code, "1");
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
