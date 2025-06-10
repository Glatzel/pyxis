use std::ptr;
use std::str::FromStr;

use envoy::{AsVecPtr, CStrToString, ToCString};
use miette::IntoDiagnostic;

use crate::data_types::iso19111::*;
use crate::{OPTION_NO, OPTION_YES, Proj, check_result, pj_obj_list_to_vec};
/// # ISO-19111 Base functions
impl Proj<'_> {
    ///Return the type of an object.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_type>
    pub fn get_type(&self) -> miette::Result<ProjType> {
        let result = unsafe { proj_sys::proj_get_type(self.ptr()) };
        ProjType::try_from(result).into_diagnostic()
    }
    ///Return whether an object is deprecated.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_deprecated>
    pub fn is_deprecated(&self) -> bool { unsafe { proj_sys::proj_is_deprecated(self.ptr()) != 0 } }
    ///Return a list of non-deprecated objects related to the passed one.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_non_deprecated>
    pub fn get_non_deprecated(&self) -> miette::Result<Vec<Proj<'_>>> {
        let result = unsafe { proj_sys::proj_get_non_deprecated(self.ctx.ptr, self.ptr()) };
        pj_obj_list_to_vec(self.ctx, result)
    }
    ///Return whether two objects are equivalent.
    ///
    /// # Argument
    ///
    /// * `other`: Other object
    /// * `criterion`: Comparison criterion
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_equivalent_to>
    pub fn is_equivalent_to(&self, other: &Proj, criterion: ComparisonCriterion) -> bool {
        unsafe { proj_sys::proj_is_equivalent_to(self.ptr(), other.ptr(), criterion.into()) != 0 }
    }

    /// Return whether two objects are equivalent.
    ///
    ///Possibly using database to check for name aliases.
    ///
    /// # Argument
    ///
    /// * `other`: Other object
    /// * `criterion`: Comparison criterion
    ///
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
    ///Return whether an object is a CRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_crs>
    pub fn is_crs(&self) -> bool { unsafe { proj_sys::proj_is_crs(self.ptr()) != 0 } }
    /// Get the name of an object.
    ///
    ///The lifetime of the returned string is the same as the input obj
    /// parameter.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_name>
    pub fn get_name(&self) -> String {
        unsafe { proj_sys::proj_get_name(self.ptr()) }
            .to_string()
            .unwrap_or_default()
    }
    ///Get the authority name / codespace of an identifier of an object.
    ///
    /// # Arguments
    ///
    /// * index: Index of the identifier. 0 = first identifier
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_id_auth_name>
    pub fn get_id_auth_name(&self, index: u16) -> Option<String> {
        unsafe { proj_sys::proj_get_id_auth_name(self.ptr(), index as i32) }.to_string()
    }
    ///Get the code of an identifier of an object.
    ///
    /// # Arguments
    ///
    /// * index: Index of the identifier. 0 = first identifier
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_id_code>
    pub fn get_id_code(&self, index: u16) -> Option<String> {
        unsafe { proj_sys::proj_get_id_code(self.ptr(), index as i32) }.to_string()
    }
    ///Get the remarks of an object.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_remarks>
    pub fn get_remarks(&self) -> String {
        unsafe { proj_sys::proj_get_remarks(self.ptr()) }
            .to_string()
            .unwrap_or_default()
    }
    ///Get the number of domains/usages for a given object.
    ///
    ///Most objects have a single domain/usage, but for some of them, there
    /// might be multiple.
    ///
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
    ///Get the scope of an object.
    ///
    ///In case of multiple usages, this will be the one of first usage.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_scope>
    pub fn get_scope(&self) -> Option<String> {
        unsafe { proj_sys::proj_get_scope(self.ptr()) }.to_string()
    }
    ///Get the scope of an object.
    ///
    /// # Arguments
    ///
    /// * `domainIdx`: Index of the domain/usage. In
    ///   [0,proj_get_domain_count(obj)[
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_scope_ex>
    pub fn get_scope_ex(&self, domain_idx: u16) -> Option<String> {
        unsafe { proj_sys::proj_get_scope_ex(self.ptr(), domain_idx as i32) }.to_string()
    }

    ///Return the area of use of an object.
    ///
    ///In case of multiple usages, this will be the one of first usage.
    ///
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
    ///Return the area of use of an object.
    ///
    /// # Arguments
    ///
    /// * `domainIdx`: Index of the domain/usage. In
    ///   [0,proj_get_domain_count(obj)[
    ///
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
    /// Get a WKT representation of an object.
    ///
    ///The returned string is valid while the input obj parameter is valid, and
    /// until a next call to proj_as_wkt() with the same input object.
    ///
    /// # Arguments
    ///
    /// * `wkt_type`: WKT version.
    /// * `multiline`:Defaults to `true`, except for WKT1_ESRI
    /// * `indentation_width`: number. Defaults to 4 (when multiline output is
    ///   on).
    /// * `output_axis`: In AUTO mode, axis will be output for WKT2 variants,
    ///   for WKT1_GDAL for ProjectedCRS with easting/northing ordering
    ///   (otherwise stripped), but not for WKT1_ESRI. Setting to `true` will
    ///   output them unconditionally, and to `false` will omit them
    ///   unconditionally.
    /// * `strict`: Default is `true`. If NO, a Geographic 3D CRS can be for
    ///   example exported as WKT1_GDAL with 3 axes, whereas this is normally
    ///   not allowed.
    /// * `ALLOW_ELLIPSOIDAL_HEIGHT_AS_VERTICAL_CRS`: Default is `false`. If set
    ///   to `true` and type == PJ_WKT1_GDAL, a Geographic 3D CRS or a Projected
    ///   3D CRS will be exported as a compound CRS whose vertical part
    ///   represents an ellipsoidal height (for example for use with LAS 1.4
    ///   WKT1).
    /// * `allow_linunit_node`: Default is `true` starting with PROJ 9.1. Only
    ///   taken into account with type == PJ_WKT1_ESRI on a Geographic 3D CRS.
    ///
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
        let result = unsafe {
            proj_sys::proj_as_wkt(
                self.ctx.ptr,
                self.ptr(),
                wkt_type.into(),
                options.as_vec_ptr().as_ptr(),
            )
        }
        .to_string();
        check_result!(self);
        Ok(result.expect("Error"))
    }
    ///Get a PROJ string representation of an object.
    ///
    /// # Arguments
    ///
    /// * `string_type`: PROJ String version.
    /// * `use_approx_tmerc`: `true` to add the +approx flag to +proj=tmerc or
    ///   +proj=utm.
    /// * `multiline`: Defaults to NO
    /// * `indentation_width`: Defaults to 2 (when multiline output is on).
    /// * `max_line_length`: Defaults to 80 (when multiline output is on).
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_as_proj_string>
    pub fn as_proj_string(
        &self,
        string_type: ProjStringType,
        use_approx_tmerc: bool,
        multiline: Option<bool>,
        indentation_width: Option<usize>,
        max_line_length: Option<usize>,
    ) -> miette::Result<String> {
        let mut options = crate::ProjOptions::new(6);
        if use_approx_tmerc {
            options.push(use_approx_tmerc, "USE_APPROX_TMERC");
        }
        options
            .push_optional(multiline, "MULTILINE", OPTION_NO)
            .push_optional(indentation_width, "INDENTATION_WIDTH", "2")
            .push_optional(max_line_length, "MAX_LINE_LENGTH", "80");

        let result = unsafe {
            proj_sys::proj_as_proj_string(
                self.ctx.ptr,
                self.ptr(),
                string_type.into(),
                options.as_vec_ptr().as_ptr(),
            )
        }
        .to_string();
        check_result!(self);
        Ok(result.expect("Error"))
    }
    ///Get a PROJJSON string representation of an object.
    ///
    /// # Arguments
    ///
    /// * `multiline`: Defaults to `true`
    /// * `indentation_width`: Defaults to 2 (when multiline output is on).
    /// * `schema`: URL to PROJJSON schema. Can be set to empty string to
    ///   disable it.
    ///
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

        let result = unsafe {
            proj_sys::proj_as_projjson(self.ctx.ptr, self.ptr(), options.as_vec_ptr().as_ptr())
        }
        .to_string();
        check_result!(self);
        Ok(result.expect("Error"))
    }

    ///Return the base CRS of a BoundCRS or a DerivedCRS/ProjectedCRS, or the
    /// source CRS of a CoordinateOperation, or the CRS of a CoordinateMetadata.
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_source_crs>
    pub fn get_source_crs(&self) -> Option<Proj<'_>> {
        let out_ptr = unsafe { proj_sys::proj_get_source_crs(self.ctx.ptr, self.ptr()) };
        if out_ptr.is_null() {
            return None;
        }
        Some(Self::new(self.ctx, out_ptr).unwrap())
    }
    ///Return the hub CRS of a BoundCRS or the target CRS of a
    /// CoordinateOperation.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_target_crs>
    pub fn get_target_crs(&self) -> Option<Proj<'_>> {
        let out_ptr = unsafe { proj_sys::proj_get_target_crs(self.ctx.ptr, self.ptr()) };
        if out_ptr.is_null() {
            return None;
        }
        Some(Self::new(self.ctx, out_ptr).unwrap())
    }
    ///Identify the CRS with reference CRSs.
    ///
    ///The candidate CRSs are either hard-coded, or looked in the database when
    /// it is available.
    ///
    ///Note that the implementation uses a set of heuristics to have a good
    /// compromise of successful identifications over execution time. It might
    /// miss legitimate matches in some circumstances.
    ///
    ///The method returns a list of matching reference CRS, and the percentage
    /// (0-100) of confidence in the match. The list is sorted by decreasing
    /// confidence.
    ///
    /// * 100% means that the name of the reference entry perfectly matches the
    ///   CRS name, and both are equivalent. In which case a single result is
    ///   returned. Note: in the case of a GeographicCRS whose axis order is
    ///   implicit in the input definition (for example ESRI WKT), then axis
    ///   order is ignored for the purpose of identification. That is the CRS
    ///   built from
    ///   GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.
    ///   0, 298.257223563]],
    ///   PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]] will be
    ///   identified to EPSG:4326, but will not pass a isEquivalentTo(EPSG_4326,
    ///   util::IComparable::Criterion::EQUIVALENT) test, but rather
    ///   isEquivalentTo(EPSG_4326,
    ///   util::IComparable::Criterion::EQUIVALENT_EXCEPT_AXIS_ORDER_GEOGCRS)
    /// * 90% means that CRS are equivalent, but the names are not exactly the
    ///   same.
    /// * 70% means that CRS are equivalent, but the names are not equivalent.
    /// * 25% means that the CRS are not equivalent, but there is some
    ///   similarity in the names.
    ///
    ///Other confidence values may be returned by some specialized
    /// implementations.
    ///
    /// This is implemented for GeodeticCRS, ProjectedCRS,
    /// VerticalCRS and CompoundCRS. Return the hub CRS of a BoundCRS or the
    /// target CRS of a CoordinateOperation.
    ///
    /// # Arguments
    ///
    /// * `auth_name`: Authority name, or NULL for all authorities
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_identify>
    pub fn identify(&self, auth_name: &str) -> miette::Result<Vec<Proj<'_>>> {
        let mut confidence: Vec<i32> = Vec::new();
        let result = unsafe {
            proj_sys::proj_identify(
                self.ctx.ptr,
                self.ptr(),
                auth_name.to_cstring().as_ptr(),
                ptr::null(),
                &mut confidence.as_mut_ptr(),
            )
        };
        pj_obj_list_to_vec(self.ctx, result)
    }
    ///Suggests a database code for the passed object.
    ///
    ///Supported type of objects are PrimeMeridian, Ellipsoid, Datum,
    /// DatumEnsemble, GeodeticCRS, ProjectedCRS, VerticalCRS, CompoundCRS,
    /// BoundCRS, Conversion.
    ///
    /// # Arguments
    ///
    /// * `authority`: Authority name into which the object will be inserted.
    /// * `numeric_code`: Whether the code should be numeric, or derived from
    ///   the object name.
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_suggests_code_for>
    pub fn suggests_code_for(&self, authority: &str, numeric_code: bool) -> String {
        let result = unsafe {
            proj_sys::proj_suggests_code_for(
                self.ctx.ptr,
                self.ptr(),
                authority.to_cstring().as_ptr(),
                numeric_code as i32,
                ptr::null(),
            )
        };
        result.to_string().expect("Error")
    }
    ///Returns whether a CRS is a derived CRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_is_derived>
    pub fn crs_is_derived(&self) -> bool {
        unsafe { proj_sys::proj_crs_is_derived(self.ctx.ptr, self.ptr()) != 0 }
    }
    ///Get the geodeticCRS / geographicCRS from a CRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_geodetic_crs>
    pub fn crs_get_geodetic_crs(&self) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_crs_get_geodetic_crs(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    /// Get the horizontal datum from a CRS.
    ///
    /// This function may return a Datum or DatumEnsemble object.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_horizontal_datum>
    pub fn crs_get_horizontal_datum(&self) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_crs_get_horizontal_datum(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Get a CRS component from a CompoundCRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_sub_crs>
    pub fn crs_get_sub_crs(&self, index: u16) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_crs_get_sub_crs(self.ctx.ptr, self.ptr(), index as i32) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Returns the datum of a SingleCRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_datum>
    pub fn crs_get_datum(&self) -> miette::Result<Option<Proj<'_>>> {
        let ptr = unsafe { proj_sys::proj_crs_get_datum(self.ctx.ptr, self.ptr()) };
        check_result!(self);
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(crate::Proj::new(self.ctx, ptr).unwrap()))
    }
    /// Returns the datum ensemble of a SingleCRS.
    ///
    ///This function may return a Datum or DatumEnsemble object.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_datum_ensemble>
    pub fn crs_get_datum_ensemble(&self) -> miette::Result<Option<Proj<'_>>> {
        let ptr = unsafe { proj_sys::proj_crs_get_datum_ensemble(self.ctx.ptr, self.ptr()) };
        check_result!(self);
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(crate::Proj::new(self.ctx, ptr).unwrap()))
    }
    ///Returns a datum for a SingleCRS.
    ///
    ///If the SingleCRS has a datum, then this datum is returned. Otherwise,
    /// the SingleCRS has a datum ensemble, and this datum ensemble is returned
    /// as a regular datum instead of a datum ensemble.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_datum_forced>
    pub fn crs_get_datum_forced(&self) -> miette::Result<Option<Proj<'_>>> {
        let ptr = unsafe { proj_sys::proj_crs_get_datum_forced(self.ctx.ptr, self.ptr()) };
        check_result!(self);
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(crate::Proj::new(self.ctx, ptr).unwrap()))
    }
    ///Return whether a CRS has an associated PointMotionOperation.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_has_point_motion_operation>
    pub fn crs_has_point_motion_operation(&self) -> bool {
        unsafe { proj_sys::proj_crs_has_point_motion_operation(self.ctx.ptr, self.ptr()) != 0 }
    }
    ///Return whether a CRS has an associated PointMotionOperation.
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_datum_ensemble_get_member_count>
    pub fn datum_ensemble_get_member_count(&self) -> u16 {
        unsafe { proj_sys::proj_datum_ensemble_get_member_count(self.ctx.ptr, self.ptr()) as u16 }
    }
    /// Returns the positional accuracy of the datum ensemble.
    ///
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
    ///Returns a member from a datum ensemble.
    ///
    /// # Arguments
    /// * member_index: Index of the datum member to extract (between 0 and
    ///   proj_datum_ensemble_get_member_count()-1)
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_datum_ensemble_get_member>
    pub fn datum_ensemble_get_member(&self, member_index: u16) -> miette::Result<Option<Proj<'_>>> {
        let ptr = unsafe {
            proj_sys::proj_datum_ensemble_get_member(self.ctx.ptr, self.ptr(), member_index as i32)
        };
        check_result!(self);
        if ptr.is_null() {
            return Ok(None);
        }
        Ok(Some(crate::Proj::new(self.ctx, ptr).unwrap()))
    }
    ///Returns the frame reference epoch of a dynamic geodetic or vertical
    ///
    /// reference frame.
    ///
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
    //Returns the coordinate system of a SingleCRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_coordinate_system>
    pub fn crs_get_coordinate_system(&self) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_crs_get_coordinate_system(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Returns the type of the coordinate system.
    ///
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
    ///Returns the number of axis of the coordinate system.
    ///
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
    ///Returns information on an axis.
    ///
    /// # Arguments
    /// * `index`: Index of the coordinate system (between 0 and
    ///   proj_cs_get_axis_count() - 1)
    ///
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
    ///Get the ellipsoid from a CRS or a GeodeticReferenceFrame.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_ellipsoid>
    pub fn get_ellipsoid(&self) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_get_ellipsoid(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Return ellipsoid parameters.
    ///
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
    ///Get the name of the celestial body of this object.
    ///
    ///Object should be a CRS, Datum or Ellipsoid.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_celestial_body_name>
    pub fn get_celestial_body_name(&self) -> Option<String> {
        unsafe { proj_sys::proj_get_celestial_body_name(self.ctx.ptr, self.ptr()) }.to_string()
    }
    ///Get the prime meridian of a CRS or a GeodeticReferenceFrame.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_prime_meridian>
    pub fn get_prime_meridian(&self) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_get_prime_meridian(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Return prime meridian parameters.
    ///
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
    ///Return the Conversion of a DerivedCRS (such as a ProjectedCRS), or the
    /// Transformation from the baseCRS to the hubCRS of a BoundCRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_get_coordoperation>
    pub fn crs_get_coordoperation(&self) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_crs_get_coordoperation(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Return information on the operation method of the SingleOperation.
    ///
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
    ///Return whether a coordinate operation can be instantiated as a PROJ
    /// pipeline, checking in particular that referenced grids are available.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_is_instantiable>
    pub fn coordoperation_is_instantiable(&self) -> bool {
        unsafe { proj_sys::proj_coordoperation_is_instantiable(self.ctx.ptr, self.ptr()) != 0 }
    }
    ///Return whether a coordinate operation has a "ballpark" transformation,
    /// that is a very approximate one, due to lack of more accurate
    /// transformations.
    ///
    ///Typically a null geographic offset between two horizontal datum, or a
    /// null vertical offset (or limited to unit changes) between two vertical
    /// datum. Errors of several tens to one hundred meters might be expected,
    /// compared to more accurate transformations.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_has_ballpark_transformation>
    pub fn coordoperation_has_ballpark_transformation(&self) -> bool {
        unsafe {
            proj_sys::proj_coordoperation_has_ballpark_transformation(self.ctx.ptr, self.ptr()) != 0
        }
    }
    ///Return whether a coordinate operation requires coordinate tuples to have
    /// a valid input time for the coordinate transformation to succeed. (this
    /// applies for the forward direction)
    ///
    ///Note: in the case of a time-dependent Helmert transformation, this
    /// function will return true, but when executing proj_trans(), execution
    /// will still succeed if the time information is missing, due to the
    /// transformation central epoch being used as a fallback.
    ///
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
    ///Return the number of parameters of a SingleOperation.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_param_count>
    pub fn coordoperation_get_param_count(&self) -> u16 {
        unsafe { proj_sys::proj_coordoperation_get_param_count(self.ctx.ptr, self.ptr()) as u16 }
    }
    ///Return the index of a parameter of a SingleOperation.
    ///
    /// # Arguments
    /// * `name`: Parameter name.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_param_index>
    pub fn coordoperation_get_param_index(&self, name: &str) -> miette::Result<u16> {
        let result = unsafe {
            proj_sys::proj_coordoperation_get_param_index(
                self.ctx.ptr,
                self.ptr(),
                name.to_cstring().as_ptr(),
            )
        };
        if result == -1 {
            miette::bail!("Error");
        }
        Ok(result as u16)
    }
    ///Return a parameter of a SingleOperation.
    ///
    /// # Arguments
    ///
    /// * `index`: Parameter index.
    ///
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
    ///Return the number of grids used by a CoordinateOperation.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_get_grid_used_count>
    pub fn coordoperation_get_grid_used_count(&self) -> u16 {
        unsafe {
            proj_sys::proj_coordoperation_get_grid_used_count(self.ctx.ptr, self.ptr()) as u16
        }
    }
    ///Return a parameter of a SingleOperation.
    ///
    /// # Arguments
    ///
    /// * `index`: Parameter index.
    ///
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
    ///Return the accuracy (in metre) of a coordinate operation.
    ///
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
    ///Return the parameters of a Helmert transformation as WKT1 TOWGS84
    /// values.
    ///
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
    ///Returns a PJ* coordinate operation object which represents the inverse
    /// operation of the specified coordinate operation.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordoperation_create_inverse>
    pub fn coordoperation_create_inverse(&self) -> miette::Result<Proj<'_>> {
        let ptr = unsafe { proj_sys::proj_coordoperation_create_inverse(self.ctx.ptr, self.ptr()) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Returns the number of steps of a concatenated operation.
    ///
    ///The input object must be a concatenated operation.
    ///
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
    ///Returns a step of a concatenated operation.
    ///
    ///The input object must be a concatenated operation.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_concatoperation_get_step>
    pub fn concatoperation_get_step(&self, index: u16) -> miette::Result<Proj<'_>> {
        let ptr = unsafe {
            proj_sys::proj_concatoperation_get_step(self.ctx.ptr, self.ptr(), index as i32)
        };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Instantiate a CoordinateMetadata object.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordinate_metadata_create>
    pub fn coordinate_metadata_create(&self, epoch: f64) -> miette::Result<Proj<'_>> {
        let ptr =
            unsafe { proj_sys::proj_coordinate_metadata_create(self.ctx.ptr, self.ptr(), epoch) };
        crate::Proj::new(self.ctx, ptr)
    }
    ///Return the coordinate epoch associated with a CoordinateMetadata.
    ///
    /// It may return a NaN value if there is no associated coordinate epoch.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_coordinate_metadata_get_epoch>
    pub fn coordinate_metadata_get_epoch(&self) -> f64 {
        unsafe { proj_sys::proj_coordinate_metadata_get_epoch(self.ctx.ptr, self.ptr()) }
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
        let proj_string = pj.as_proj_string(ProjStringType::Proj4, true, None, None, None)?;
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
        let code = crs.suggests_code_for("HOBU", true);
        assert_eq!(code, "1");
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
        let ctx = crate::new_test_ctx()?;
        let factory = ctx.create_operation_factory_context(None);
        let source_crs = ctx.create_from_database("EPSG", "28356", Category::Crs, false)?;
        let target_crs = ctx.create_from_database("EPSG", "7856", Category::Crs, false)?;
        let ops = factory
            .set_spatial_criterion(SpatialCriterion::PartialIntersection)
            .set_grid_availability_use(GridAvailabilityUse::Ignored)
            .create_operations(&source_crs, &target_crs)?;
        let op = ops.first().unwrap();
        let count = op.concatoperation_get_step_count()?;
        assert_eq!(count, 3);
        Ok(())
    }
    #[test]
    fn test_concatoperation_get_step() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let factory = ctx.create_operation_factory_context(None);
        let source_crs = ctx.create_from_database("EPSG", "28356", Category::Crs, false)?;
        let target_crs = ctx.create_from_database("EPSG", "7856", Category::Crs, false)?;
        let ops = factory
            .set_spatial_criterion(SpatialCriterion::PartialIntersection)
            .set_grid_availability_use(GridAvailabilityUse::Ignored)
            .create_operations(&source_crs, &target_crs)?;
        let op = ops.first().unwrap();
        let step = op.concatoperation_get_step(1)?;
        let wkt = step.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("GDA94 to GDA2020"));
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
