use std::ptr;

use envoy::ToCStr;

use crate::data_types::iso19111::*;
use crate::{Context, Proj, ProjOptions, pj_obj_list_to_vec};
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
    pub fn create_conversion_pole_rotation_netcdf_cf_convention(
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
    #[test]
    fn test_crs_create_bound_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let crs = ctx.create_from_database("EPSG", "4807", Category::Crs, false)?;
        let res = crs.crs_create_bound_crs_to_wgs84(None)?;
        let base_crs = res.get_source_crs().unwrap();
        let hub_crs = res.get_target_crs().unwrap();
        let transf = res.crs_get_coordoperation()?;
        let bound_crs = ctx.crs_create_bound_crs(&base_crs, &hub_crs, &transf)?;
        let wkt = bound_crs.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_crs_create_bound_vertical_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let crs = ctx.create("EPSG:4979")?;
        let vert_crs =
            ctx.create_vertical_crs(Some("myVertCRS"), Some("myVertDatum"), None, 0.0)?;

        let bound_crs = ctx.crs_create_bound_vertical_crs(&vert_crs, &crs, "foo.gtx")?;
        let wkt = bound_crs.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_utm() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_utm(31, true)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        assert!(wkt.contains("UTM zone 31N"));
        Ok(())
    }
    #[test]
    fn test_create_conversion_transverse_mercator() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_transverse_mercator(
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_gauss_schreiber_transverse_mercator() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_gauss_schreiber_transverse_mercator(
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_transverse_mercator_south_oriented() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_transverse_mercator_south_oriented(
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_two_point_equidistant() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_two_point_equidistant(
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_tunisia_mapping_grid() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_tunisia_mapping_grid(
            0.0,
            0.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_tunisia_mining_grid() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_tunisia_mining_grid(
            0.0,
            0.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_albers_equal_area() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_albers_equal_area(
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_1sp() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_1sp(
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_1sp_variant_b() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_1sp_variant_b(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_2sp() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_2sp(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_2sp_michigan() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_2sp_michigan(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_2sp_belgium() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_2sp_belgium(
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_azimuthal_equidistant() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_azimuthal_equidistant(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_guam_projection() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_guam_projection(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_createconversion_bonne() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_bonne(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_cylindrical_equal_area_spherical() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_cylindrical_equal_area_spherical(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_cylindrical_equal_area() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_cylindrical_equal_area(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_cassini_soldner() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_cassini_soldner(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_equidistant_conic() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_equidistant_conic(
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_i() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_i(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_ii() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_ii(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_iii() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_iii(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_iv() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_iv(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_v() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_v(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_vi() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_vi(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_equidistant_cylindrical() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_equidistant_cylindrical(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_equidistant_cylindrical_spherical() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_equidistant_cylindrical_spherical(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_gall() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_gall(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_goode_homolosine() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_goode_homolosine(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_interrupted_goode_homolosine() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_interrupted_goode_homolosine(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_geostationary_satellite_sweep_x() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_geostationary_satellite_sweep_x(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_geostationary_satellite_sweep_y() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_geostationary_satellite_sweep_y(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_gnomonic() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_gnomonic(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_hotine_oblique_mercator_variant_a() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_hotine_oblique_mercator_variant_a(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_hotine_oblique_mercator_variant_b() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_hotine_oblique_mercator_variant_b(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_hotine_oblique_mercator_two_point_natural_origin()
    -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_hotine_oblique_mercator_two_point_natural_origin(
            1.0,
            1.0,
            1.0,
            2.0,
            3.0,
            1.0,
            4.0,
            6.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_laborde_oblique_mercator() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_laborde_oblique_mercator(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_international_map_world_polyconic() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_international_map_world_polyconic(
            1.0,
            2.0,
            4.0,
            5.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_krovak_north_oriented() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_krovak_north_oriented(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_krovak() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_krovak(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_azimuthal_equal_area() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_azimuthal_equal_area(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_miller_cylindrical() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_miller_cylindrical(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_mercator_variant_a() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_mercator_variant_a(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_mercator_variant_b() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_mercator_variant_b(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_popular_visualisation_pseudo_mercator() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_popular_visualisation_pseudo_mercator(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_mollweide() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_mollweide(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_new_zealand_mapping_grid() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_new_zealand_mapping_grid(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_oblique_stereographic() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_oblique_stereographic(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_orthographic() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_orthographic(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_local_orthographic() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_local_orthographic(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_american_polyconic() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_american_polyconic(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_polar_stereographic_variant_a() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_polar_stereographic_variant_a(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_polar_stereographic_variant_b() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_polar_stereographic_variant_b(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_robinson() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_robinson(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_sinusoidal() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_sinusoidal(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_stereographic() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_stereographic(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_van_der_grinten() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_van_der_grinten(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_i() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_i(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_ii() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_ii(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_iii() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_iii(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_iv() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_iv(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_v() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_v(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_vi() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_vi(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_vii() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_vii(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_quadrilateralized_spherical_cube() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_quadrilateralized_spherical_cube(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_spherical_cross_track_height() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_spherical_cross_track_height(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }

    #[test]
    fn test_create_conversion_equal_earth() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_equal_earth(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_vertical_perspective() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_vertical_perspective(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_pole_rotation_grib_convention() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_pole_rotation_grib_convention(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
    #[test]
    fn test_create_conversion_pole_rotation_netcdf_cf_convention() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_pole_rotation_netcdf_cf_convention(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{}", wkt);
        Ok(())
    }
}
