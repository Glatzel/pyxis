use envoy::ToCStr;

use crate::{Proj, check_result};
/// # Transformation setup
///
///The objects returned by the functions defined in this section have minimal
/// interaction with the functions of the C API for ISO-19111 functionality, and
/// vice versa. See its introduction paragraph for more details.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c-api-for-iso-19111-functionality>
impl crate::Context {
    ///Create a transformation object, or a CRS object, from:
    ///
    /// * a proj-string
    /// * a WKT string
    /// * an object code (like "EPSG:4326", "urn:ogc:def:crs:EPSG::4326"
    /// * "urn:ogc:def:coordinateOperation:EPSG::1671")
    /// * an Object name. e.g "WGS 84", "WGS 84 / UTM zone 31N". In that case as
    /// uniqueness is not guaranteed, heuristics are applied to determine the
    /// appropriate best match.
    /// * a OGC URN combining references for compound coordinate reference
    ///   systems
    /// (e.g "urn:ogc:def:crs,crs:EPSG::2393,crs:EPSG::5717" or custom
    /// abbreviated syntax "EPSG:2393+5717"),
    /// * a OGC URN combining references for concatenated operations (e.g.
    /// "urn:ogc:def:coordinateOperation,coordinateOperation:EPSG::3895,
    /// coordinateOperation:EPSG::1618")
    /// * a PROJJSON string. The jsonschema is at https://proj.org/schemas/v0.4/projjson.schema.json
    ///   (added in 6.2)
    /// * a compound CRS made from two object names separated with " + ". e.g.
    /// "WGS 84 + EGM96 height" (added in 7.1)
    ///
    /// If a proj-string contains a +type=crs option, then it is interpreted as
    /// a CRS definition. In particular geographic CRS are assumed to have axis
    /// in the longitude, latitude order and with degree angular unit. The use
    /// of proj-string to describe a CRS is discouraged. It is a legacy means of
    /// conveying CRS descriptions: use of object codes (EPSG:XXXX typically) or
    /// WKT description is recommended for better expressivity.
    ///
    /// If a proj-string does not contain +type=crs, then it is interpreted as a
    /// coordination operation / transformation.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create>
    pub fn create(&self, definition: &str) -> miette::Result<crate::Proj> {
        let ptr = unsafe { proj_sys::proj_create(self.ptr, definition.to_cstr()) };
        check_result!(self);
        Proj::new(self, ptr)
    }
    ///Create a transformation object, or a CRS object, with argc/argv-style
    /// initialization. For this application each parameter in the defining
    /// proj-string is an entry in argv.
    ///
    /// If there is a type=crs argument, then the arguments are interpreted as a
    /// CRS definition. In particular geographic CRS are assumed to have axis in
    /// the longitude, latitude order and with degree angular unit.
    ///
    /// If there is no type=crs argument, then it is interpreted as a
    /// coordination operation / transformation.
    ///
    ///  # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv>
    pub fn create_argv(&self, argv: &[&str]) -> miette::Result<crate::Proj> {
        let len = argv.len();
        let mut argv_ptrs: Vec<*mut i8> = Vec::with_capacity(len);
        for s in argv {
            argv_ptrs.push((*s).to_cstring().into_raw());
        }
        let ptr =
            unsafe { proj_sys::proj_create_argv(self.ptr, len as i32, argv_ptrs.as_mut_ptr()) };
        check_result!(self);
        Proj::new(self, ptr)
    }
    ///Create a transformation object that is a pipeline between two known
    /// coordinate reference systems.
    ///
    /// source_crs and target_crs can be :
    ///
    /// * a "AUTHORITY:CODE", like EPSG:25832. When using that syntax for a
    ///   source CRS, the created pipeline will expect that the values passed to
    ///   proj_trans() respect the axis order and axis unit of the official
    ///   definition ( so for example, for EPSG:4326, with latitude first and
    ///   longitude next, in degrees). Similarly, when using that syntax for a
    ///   target CRS, output values will be emitted according to the official
    ///   definition of this CRS.
    /// * a PROJ string, like "+proj=longlat +datum=WGS84". When using that
    /// syntax, the axis order and unit for geographic CRS will be longitude,
    /// latitude, and the unit degrees.
    /// * the name of a CRS as found in the PROJ database, e.g "WGS84", "NAD27",
    ///   etc.
    /// * more generally any string accepted by proj_create() representing a CRS
    ///
    /// Starting with PROJ 9.2, source_crs (exclusively) or target_crs can be a
    /// CoordinateMetadata with an associated coordinate epoch.
    ///
    ///Starting with PROJ 9.4, both source_crs and target_crs can be a
    /// CoordinateMetadata with an associated coordinate epoch, to perform
    /// changes of coordinate epochs. Note however than this is in practice
    /// limited to use of velocity grids inside the same dynamic CRS.
    ///
    ///An "area of use" can be specified in area. When it is supplied, the more
    /// accurate transformation between two given systems can be chosen.
    ///
    ///When no area of use is specific and several coordinate operations are
    /// possible depending on the area of use, this function will internally
    /// store those candidate coordinate operations in the return PJ object.
    /// Each subsequent coordinate transformation done with proj_trans() will
    /// then select the appropriate coordinate operation by comparing the input
    /// coordinates with the area of use of the candidate coordinate operations.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs>
    pub fn create_crs_to_crs(
        &self,
        source_crs: &str,
        target_crs: &str,
        area: &crate::Area,
    ) -> miette::Result<crate::Proj> {
        let ptr = unsafe {
            proj_sys::proj_create_crs_to_crs(
                self.ptr,
                source_crs.to_cstr(),
                target_crs.to_cstr(),
                area.ptr,
            )
        };
        check_result!(self);
        Proj::new(self, ptr)
    }
    ///Added in version 6.2.0.
    ///
    ///Create a transformation object that is a pipeline between two known
    /// coordinate reference systems.
    ///
    ///This is the same as proj_create_crs_to_crs() except that the source and
    /// target CRS are passed as PJ* objects which must be of the CRS variety.
    ///
    ///Starting with PROJ 9.2, source_crs (exclusively) or target_crs can be a
    /// CoordinateMetadata with an associated coordinate epoch.
    ///
    ///Starting with PROJ 9.4, both source_crs and target_crs can be a
    /// CoordinateMetadata with an associated coordinate epoch, to perform
    /// changes of coordinate epochs. Note however than this is in practice
    /// limited to use of velocity grids inside the same dynamic CRS.
    ///
    ///
    /// #Parameters
    ///
    /// * authority: to restrict the authority of coordinate operations looked
    ///   up in the database. When not specified, coordinate operations from any
    ///   authority will be searched, with the restrictions set in the
    ///   authority_to_authority_preference database table related to the
    ///   authority of the source/target CRS themselves. If authority is set to
    ///   any, then coordinate operations from any authority will be searched.
    ///   If authority is a non-empty string different of any, then coordinate
    ///   operations will be searched only in that authority namespace (e.g
    ///   EPSG).
    /// * accuracy: to set the minimum desired accuracy (in metres) of the
    ///   candidate coordinate operations.
    /// * allow_ballpark can be set to NO to disallow the use of Ballpark
    ///   transformation in the candidate coordinate operations.
    /// * only_best: (PROJ >= 9.2) Can be set to YES to cause PROJ to error out
    ///   if the best transformation, known of PROJ, and usable by PROJ if all
    ///   grids known and usable by PROJ were accessible, cannot be used. Best
    ///   transformation should be understood as the transformation returned by
    ///   proj_get_suggested_operation() if all known grids were accessible
    ///   (either locally or through network). Note that the default value for
    ///   this option can be also set with the PROJ_ONLY_BEST_DEFAULT
    ///   environment variable, or with the only_best_default setting of
    ///   proj.ini (the ONLY_BEST option when specified overrides such default
    ///   value).
    /// * force_over=YES/NO: can be set to YES to force the +over flag on the
    ///   transformation returned by this function. See Longitude Wrapping
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs_from_pj>
    pub fn create_crs_to_crs_from_pj(
        &self,
        source_crs: crate::Proj,
        target_crs: crate::Proj,
        area: &crate::Area,
        authority: Option<&str>,
        accuracy: Option<f64>,
        allow_ballpark: Option<bool>,
        only_best: Option<bool>,
        force_over: Option<bool>,
    ) -> miette::Result<crate::Proj> {
        let mut options = crate::ProjOptions::new(5);
        options
            .push_optional_pass(authority, "AUTHORITY")
            .push_optional_pass(accuracy, "ACCURACY")
            .push_optional_pass(allow_ballpark, "ALLOW_BALLPARK")
            .push_optional_pass(only_best, "ONLY_BEST")
            .push_optional_pass(force_over, "FORCE_OVER");
        let ptrs = options.vec_ptr();
        let ptr = unsafe {
            proj_sys::proj_create_crs_to_crs_from_pj(
                self.ptr,
                source_crs.ptr(),
                target_crs.ptr(),
                area.ptr,
                ptrs.as_ptr(),
            )
        };
        check_result!(self);
        Proj::new(self, ptr)
    }
    ///Returns a PJ* object whose axis order is the one expected for
    /// visualization purposes.
    ///
    ///The input object must be either:
    ///
    /// * a coordinate operation, that has been created with
    ///   proj_create_crs_to_crs(). If the axis order of its source or target
    ///   CRS is northing,easting, then an axis swap operation will be inserted.
    /// * a CRS. The axis order of geographic CRS will be longitude, latitude
    ///   [,height], and the one of projected CRS will be easting, northing [,
    ///   height]
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_normalize_for_visualization>
    pub fn normalize_for_visualization(&self, obj: &crate::Proj) -> miette::Result<crate::Proj> {
        let ptr = unsafe { proj_sys::proj_normalize_for_visualization(self.ptr, obj.ptr()) };
        Proj::new(self, ptr)
    }
}

impl Drop for crate::Proj<'_> {
    ///Deallocate a PJ transformation object.
    ///
    /// # References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_destroy>
    fn drop(&mut self) { unsafe { proj_sys::proj_destroy(self.ptr()) }; }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_create() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let _ = pj.clone();
        Ok(())
    }

    #[test]
    fn test_create_argv() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let _ = ctx.create_argv(&["proj=utm", "zone=32", "ellps=GRS80"])?;
        Ok(())
    }

    #[test]
    fn test_create_crs_to_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let _ = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::Area::default())?;
        Ok(())
    }

    #[test]
    fn test_create_crs_to_crs_from_pj() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj1 = ctx.create("EPSG:4326")?;
        let pj2 = ctx.create("EPSG:4978")?;
        let _ = ctx.create_crs_to_crs_from_pj(
            pj1,
            pj2,
            &crate::Area::default(),
            Some("any"),
            Some(0.001),
            Some(true),
            Some(true),
            Some(true),
        )?;
        Ok(())
    }
}
