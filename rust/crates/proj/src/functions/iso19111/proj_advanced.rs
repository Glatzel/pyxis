use std::ptr;

use envoy::{AsVecPtr, ToCString};

use crate::data_types::iso19111::*;
use crate::{Proj, ProjOptions};
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
    pub fn alter_name(&self, name: &str) -> miette::Result<Proj<'_>> {
        let ptr = unsafe {
            proj_sys::proj_alter_name(self.ctx.ptr, self.ptr(), name.to_cstring().as_ptr())
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_alter_id>
    pub fn alter_id(&self, auth_name: &str, code: &str) -> miette::Result<Proj<'_>> {
        let ptr = unsafe {
            proj_sys::proj_alter_id(
                self.ctx.ptr,
                self.ptr(),
                auth_name.to_cstring().as_ptr(),
                code.to_cstring().as_ptr(),
            )
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_alter_geodetic_crs>
    pub fn crs_alter_geodetic_crs(&self, new_geod_crs: &Proj) -> miette::Result<Proj<'_>> {
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
    ) -> miette::Result<Proj<'_>> {
        let ptr = unsafe {
            // let angular_unit = angular_unit.map(|s| s.to_cstring());
            // let unit_auth_name = unit_auth_name.map(|s| s.to_cstring());
            // let unit_code = unit_code.map(|s| s.to_cstring());
            proj_sys::proj_crs_alter_cs_angular_unit(
                self.ctx.ptr,
                self.ptr(),
                angular_unit.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
                angular_units_convs,
                unit_auth_name.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
                unit_code.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
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
    ) -> miette::Result<Proj<'_>> {
        let ptr = unsafe {
            proj_sys::proj_crs_alter_cs_linear_unit(
                self.ctx.ptr,
                self.ptr(),
                linear_units.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
                linear_units_conv,
                unit_auth_name.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
                unit_code.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
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
    ) -> miette::Result<Proj<'_>> {
        let ptr = unsafe {
            proj_sys::proj_crs_alter_parameters_linear_unit(
                self.ctx.ptr,
                self.ptr(),
                linear_units.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
                linear_units_conv,
                unit_auth_name.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
                unit_code.map_or(ptr::null(), |s| s.to_cstring().into_raw()),
                convert_to_new_unit as i32,
            )
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_promote_to_3D>
    pub fn crs_promote_to_3d(&self, crs_3d_name: Option<&str>) -> miette::Result<Proj<'_>> {
        let crs_3d_name = crs_3d_name.map(|s| s.to_cstring());
        let ptr = unsafe {
            proj_sys::proj_crs_promote_to_3D(
                self.ctx.ptr,
                crs_3d_name.map_or(ptr::null(), |s| s.as_ptr()),
                self.ptr(),
            )
        };
        Proj::new(self.ctx, ptr)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_demote_to_2D>
    pub fn crs_demote_to_2d(&self, crs_2d_name: Option<&str>) -> miette::Result<Proj<'_>> {
        let crs_2d_name = crs_2d_name.map(|s| s.to_cstring());
        let ptr = unsafe {
            proj_sys::proj_crs_demote_to_2D(
                self.ctx.ptr,
                crs_2d_name.map_or(ptr::null(), |s| s.as_ptr()),
                self.ptr(),
            )
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
    ) -> miette::Result<Proj<'_>> {
        if new_method_epsg_code.is_none() && new_method_name.is_none() {
            miette::bail!(
                "At least one of `new_method_epsg_code` and  `new_method_name` must be set."
            )
        }
        let new_method_name = new_method_name.map(|s| s.to_cstring());
        let ptr = unsafe {
            proj_sys::proj_convert_conversion_to_other_method(
                self.ctx.ptr,
                self.ptr(),
                new_method_epsg_code.unwrap_or_default() as i32,
                new_method_name.map_or(ptr::null(), |s| s.as_ptr()),
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
    ) -> miette::Result<Proj<'_>> {
        let mut options = ProjOptions::new(1);
        options.push_optional_pass(allow_intermediate_crs, "ALLOW_INTERMEDIATE_CRS");

        let ptr = unsafe {
            proj_sys::proj_crs_create_bound_crs_to_WGS84(
                self.ctx.ptr,
                self.ptr(),
                options.as_vec_ptr().as_ptr(),
            )
        };
        crate::Proj::new(self.ctx, ptr)
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
