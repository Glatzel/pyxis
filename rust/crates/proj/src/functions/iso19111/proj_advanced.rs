use core::ptr;

use envoy::{AsVecPtr, ToCString};

use crate::data_types::iso19111::*;
use crate::{OwnedCStrings, Proj, ProjOptions};
/// # ISO-19111 Advanced functions
///
/// * <https://proj.org/en/stable/development/reference/functions.html#advanced-functions>
impl Proj {
    ///Return a copy of the object with its name changed.
    ///
    ///Currently, only implemented on CRS objects.
    ///
    /// # Arguments
    ///
    /// * `name`: New name.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_alter_name>
    pub fn alter_name(&self, name: &str) -> mischief::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_alter_name(self.ctx.ptr, self.ptr(), name.to_cstring().as_ptr())
        };
        Proj::new(&self.ctx, ptr)
    }
    ///Return a copy of the object with its identifier changed/set.
    ///
    ///Currently, only implemented on CRS objects.
    ///
    /// # Arguments
    ///
    /// * `auth_name`: Authority name.
    /// * `code`: Code.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_alter_id>
    pub fn alter_id(&self, auth_name: &str, code: &str) -> mischief::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_alter_id(
                self.ctx.ptr,
                self.ptr(),
                auth_name.to_cstring().as_ptr(),
                code.to_cstring().as_ptr(),
            )
        };
        Proj::new(&self.ctx, ptr)
    }
    ///Return a copy of the CRS with its geodetic CRS changed.
    ///
    ///Currently, when obj is a GeodeticCRS, it returns a clone of new_geod_crs
    /// When obj is a ProjectedCRS, it replaces its base CRS with new_geod_crs.
    /// When obj is a CompoundCRS, it replaces the GeodeticCRS part of the
    /// horizontal CRS with new_geod_crs. In other cases, it returns a clone of
    /// obj.
    ///
    /// # Arguments
    ///
    /// * `new_geod_crs``: Object of type GeodeticCRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_alter_geodetic_crs>
    pub fn crs_alter_geodetic_crs(&self, new_geod_crs: &Proj) -> mischief::Result<Proj> {
        let ptr = unsafe {
            proj_sys::proj_crs_alter_geodetic_crs(self.ctx.ptr, self.ptr(), new_geod_crs.ptr())
        };
        Proj::new(&self.ctx, ptr)
    }
    ///Return a copy of the CRS with its angular units changed.
    ///
    ///The CRS must be or contain a GeographicCRS.
    ///
    ///# Arguments
    /// * `angular_units`: Name of the angular units. Or `None` for Degree
    /// * `angular_units_conv`: Conversion factor from the angular unit to
    ///   radian. Or 0 for Degree if angular_units == `None`. Otherwise should
    ///   be not `None`
    /// * `unit_auth_name`: Unit authority name. Or `None`.
    /// * `unit_code`: Unit code. Or `None`.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_alter_cs_angular_unit>
    pub fn crs_alter_cs_angular_unit(
        &self,
        angular_unit: Option<&str>,
        angular_units_convs: f64,
        unit_auth_name: Option<&str>,
        unit_code: Option<&str>,
    ) -> mischief::Result<Proj> {
        let mut owned = OwnedCStrings::with_capacity(3);
        let ptr = unsafe {
            proj_sys::proj_crs_alter_cs_angular_unit(
                self.ctx.ptr,
                self.ptr(),
                owned.push_option(angular_unit),
                angular_units_convs,
                owned.push_option(unit_auth_name),
                owned.push_option(unit_code),
            )
        };
        Proj::new_with_owned_cstrings(&self.ctx, ptr, owned)
    }
    ///Return a copy of the CRS with the linear units of its coordinate system
    /// changed.
    ///
    ///The CRS must be or contain a ProjectedCRS, VerticalCRS or a
    /// GeocentricCRS.
    ///
    /// # Arguments
    ///
    /// * `linear_units`: Name of the linear units. Or `None` for Metre
    /// * `linear_units_conv`: Conversion factor from the linear unit to metre.
    ///   Or 0 for Metre if linear_units == `None`. Otherwise should be not
    ///   `None`
    /// * `unit_auth_name`: Unit authority name. Or `None`.
    /// * `unit_code`: Unit code. Or `None`.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_alter_cs_linear_unit>
    pub fn crs_alter_cs_linear_unit(
        &self,
        linear_units: Option<&str>,
        linear_units_conv: f64,
        unit_auth_name: Option<&str>,
        unit_code: Option<&str>,
    ) -> mischief::Result<Proj> {
        let mut owned = OwnedCStrings::with_capacity(3);
        let ptr = unsafe {
            proj_sys::proj_crs_alter_cs_linear_unit(
                self.ctx.ptr,
                self.ptr(),
                owned.push_option(linear_units),
                linear_units_conv,
                owned.push_option(unit_auth_name),
                owned.push_option(unit_code),
            )
        };
        Proj::new_with_owned_cstrings(&self.ctx, ptr, owned)
    }
    ///Return a copy of the CRS with the linear units of the parameters of its
    /// conversion modified.
    ///
    ///The CRS must be or contain a ProjectedCRS, VerticalCRS or a
    /// GeocentricCRS.
    ///# Arguments
    /// * `linear_units`: Name of the linear units. Or `None` for Metre
    /// * `linear_units_conv`: Conversion factor from the linear unit to metre.
    ///   Or 0 for Metre if linear_units == `None`. Otherwise should be not
    ///   `None`
    /// * `unit_auth_name`: Unit authority name. Or `None`.
    /// * `unit_code`: Unit code. Or `None`.
    /// * `convert_to_new_unit`: `true` if existing values should be converted
    ///   from their current unit to the new unit. If `false`, their value will
    ///   be left unchanged and the unit overridden (so the resulting CRS will
    ///   not be equivalent to the original one for reprojection purposes).
    ///
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
    ) -> mischief::Result<Proj> {
        let mut owned = OwnedCStrings::with_capacity(3);
        let ptr = unsafe {
            proj_sys::proj_crs_alter_parameters_linear_unit(
                self.ctx.ptr,
                self.ptr(),
                owned.push_option(linear_units),
                linear_units_conv,
                owned.push_option(unit_auth_name),
                owned.push_option(unit_code),
                convert_to_new_unit as i32,
            )
        };
        Proj::new_with_owned_cstrings(&self.ctx, ptr, owned)
    }
    ///Create a 3D CRS from an existing 2D CRS.
    ///
    ///The new axis will be ellipsoidal height, oriented upwards, and with
    /// metre units.
    ///
    /// # Arguments
    ///
    /// * `crs_3D_name`: CRS name. Or `None` (in which case the name of crs_2D
    ///   will be used)
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_promote_to_3D>
    pub fn crs_promote_to_3d(&self, crs_3d_name: Option<&str>) -> mischief::Result<Proj> {
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr = unsafe {
            proj_sys::proj_crs_promote_to_3D(
                self.ctx.ptr,
                owned.push_option(crs_3d_name),
                self.ptr(),
            )
        };
        Proj::new_with_owned_cstrings(&self.ctx, ptr, owned)
    }
    ///Create a projected 3D CRS from an existing projected 2D CRS.
    ///
    /// The passed projected_2D_crs is used so that its name is replaced by
    /// crs_name and its base geographic CRS is replaced by geog_3D_crs. The
    /// vertical axis of geog_3D_crs (ellipsoidal height) will be added as the
    /// 3rd axis of the resulting projected 3D CRS. Normally, the passed
    /// geog_3D_crs should be the 3D counterpart of the original 2D base
    /// geographic CRS of projected_2D_crs, but such no check is done.
    ///
    /// It is also possible to invoke this function with a `None` geog_3D_crs.
    /// In which case, the existing base geographic 2D CRS of
    /// projected_2D_crs will be automatically promoted to 3D by assuming a
    /// 3rd axis being an ellipsoidal height, oriented upwards, and with
    /// metre units. This is equivalent to using proj_crs_promote_to_3D().
    ///
    /// # Arguments
    ///
    /// * `crs_name`: CRS name. Or `None` (in which case the name of
    ///   projected_2D_crs will be used)
    /// * `projected_2D_crs`: Projected 2D CRS to be "promoted" to 3D.
    /// * `geog_3D_crs`: Base geographic 3D CRS for the new CRS. May be `None`.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_projected_3D_crs_from_2D>
    pub fn crs_create_projected_3d_crs_from_2d(
        &self,
        crs_name: Option<&str>,
        geog_3d_crs: Option<&Proj>,
    ) -> mischief::Result<Proj> {
        let crs_name = crs_name.map(|s| s.to_cstring());
        let ptr = unsafe {
            proj_sys::proj_crs_create_projected_3D_crs_from_2D(
                self.ctx.ptr,
                crs_name.map_or(ptr::null(), |s| s.as_ptr()),
                self.ptr(),
                geog_3d_crs.map_or(ptr::null(), |crs| crs.ptr()),
            )
        };
        Proj::new(&self.ctx, ptr)
    }
    ///Create a 2D CRS from an existing 3D CRS.
    ///
    /// # Arguments
    /// * `crs_2d_name`: CRS name. Or `None` (in which case the name of crs_3D
    ///   will be used)
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_demote_to_2D>
    pub fn crs_demote_to_2d(&self, crs_2d_name: Option<&str>) -> mischief::Result<Proj> {
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr = unsafe {
            proj_sys::proj_crs_demote_to_2D(
                self.ctx.ptr,
                owned.push_option(crs_2d_name),
                self.ptr(),
            )
        };
        Proj::new_with_owned_cstrings(&self.ctx, ptr, owned)
    }
    ///Return an equivalent projection.
    ///
    ///Currently implemented:
    ///
    /// * EPSG_CODE_METHOD_MERCATOR_VARIANT_A (1SP) to
    ///   EPSG_CODE_METHOD_MERCATOR_VARIANT_B (2SP)
    /// * EPSG_CODE_METHOD_MERCATOR_VARIANT_B (2SP) to
    ///   EPSG_CODE_METHOD_MERCATOR_VARIANT_A (1SP)
    /// * EPSG_CODE_METHOD_LAMBERT_CONIC_CONFORMAL_1SP to
    ///   EPSG_CODE_METHOD_LAMBERT_CONIC_CONFORMAL_2SP
    /// * EPSG_CODE_METHOD_LAMBERT_CONIC_CONFORMAL_2SP to
    ///   EPSG_CODE_METHOD_LAMBERT_CONIC_CONFORMAL_1SP
    ///
    /// # Arguments
    ///
    /// * `new_method_epsg_code`: EPSG code of the target method. Or 0 (in which
    ///   case new_method_name must be specified).
    /// * `new_method_name`: EPSG or PROJ target method name. Or `None` (in
    ///   which case new_method_epsg_code must be specified).
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_convert_conversion_to_other_method>
    pub fn convert_conversion_to_other_method(
        &self,
        new_method_epsg_code: Option<u16>,
        new_method_name: Option<&str>,
    ) -> mischief::Result<Proj> {
        if new_method_epsg_code.is_none() && new_method_name.is_none() {
            mischief::bail!(
                "At least one of `new_method_epsg_code` and  `new_method_name` must be set."
            )
        }
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr = unsafe {
            proj_sys::proj_convert_conversion_to_other_method(
                self.ctx.ptr,
                self.ptr(),
                new_method_epsg_code.unwrap_or_default() as i32,
                owned.push_option(new_method_name),
            )
        };
        Proj::new_with_owned_cstrings(&self.ctx, ptr, owned)
    }
    ///Returns potentially a BoundCRS, with a transformation to EPSG:4326,
    /// wrapping this CRS.
    ///
    ///# Arguments
    ///
    /// * `allow_intermediate_crs`: Defaults to NEVER. When set to
    ///   ALWAYS/IF_NO_DIRECT_TRANSFORMATION, intermediate CRS may be considered
    ///   when computing the possible transformations. Slower.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_bound_crs_to_WGS84>
    pub fn crs_create_bound_crs_to_wgs84(
        &self,
        allow_intermediate_crs: Option<AllowIntermediateCrs>,
    ) -> mischief::Result<Proj> {
        let mut options = ProjOptions::new(1);
        options.with_or_skip(allow_intermediate_crs, "ALLOW_INTERMEDIATE_CRS");

        let ptr = unsafe {
            proj_sys::proj_crs_create_bound_crs_to_WGS84(
                self.ctx.ptr,
                self.ptr(),
                options.as_vec_ptr().as_ptr(),
            )
        };
        crate::Proj::new(&self.ctx, ptr)
    }
}

#[cfg(test)]
mod test_proj_advanced {
    use strum::IntoEnumIterator;

    use super::*;
    #[test]
    fn test_alter_name() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj = ctx.create_geographic_crs(
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
    fn test_alter_id() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj = ctx.create_geographic_crs(
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
    fn test_crs_alter_geodetic_crs() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx
            .clone()
            .create_from_database("EPSG", "32631", Category::Crs, false)?;
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
    fn test_crs_alter_cs_angular_unit() -> mischief::Result<()> {
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
    fn test_crs_alter_cs_linear_unit() -> mischief::Result<()> {
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
    fn test_crs_alter_parameters_linear_unit() -> mischief::Result<()> {
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
    fn test_crs_promote_to_3d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let pj_3d = pj.crs_promote_to_3d(None)?;
        let wkt = pj_3d.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("CS[ellipsoidal,3]"));
        Ok(())
    }
    #[test]
    fn test_crs_create_projected_3d_crs_from_2d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let proj_crs = ctx
            .clone()
            .create_from_database("EPSG", "32631", Category::Crs, false)?;
        let geog_3d_crs = ctx.create_from_database("EPSG", "4979", Category::Crs, false)?;
        let pj: Proj = proj_crs.crs_create_projected_3d_crs_from_2d(None, Some(&geog_3d_crs))?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("WGS 84 / UTM zone 31N"));
        assert!(wkt.contains("CS[Cartesian,3]"));
        Ok(())
    }
    #[test]
    fn test_crs_demote_to_2d() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4979")?;
        let pj_2d = pj.crs_demote_to_2d(None)?;
        let wkt = pj_2d.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("CS[ellipsoidal,2]"));
        Ok(())
    }
    #[test]
    fn test_convert_conversion_to_other_method() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;

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
        let pj: Proj = ctx.create_projected_crs(Some("my CRS"), &geog_crs, &conv, &cs)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        let conv_in_proj = pj.crs_get_coordoperation()?;
        //by code
        {
            let new_conv = conv_in_proj.convert_conversion_to_other_method(Some(9805), None)?;
            let wkt = new_conv.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
            println!("{wkt}");
            assert!(wkt.contains("9805"));
        }
        //both none
        {
            let new_conv = conv_in_proj.convert_conversion_to_other_method(None, None);
            assert!(new_conv.is_err());
        }

        Ok(())
    }

    #[test]
    fn test_crs_create_bound_crs_to_wgs84() -> mischief::Result<()> {
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
