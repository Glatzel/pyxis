use miette::IntoDiagnostic;

use crate::data_types::iso19111::{PjComparisonCriterion, PjType};
use crate::Pj;

/// # ISO-19111
impl crate::PjContext {
    ///# References
    ///
    /// <>
    fn _context_set_autoclose_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _context_set_database_path(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _context_get_database_path(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _context_get_database_metadata(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _context_get_database_structure(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _context_guess_wkt_dialect(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_from_wkt(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _uom_get_info_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _grid_get_info_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _clone(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_from_name(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_non_deprecated(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _is_equivalent_to_with_ctx(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_area_of_use(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_area_of_use_ex(&self) { unimplemented!() }

    ///# References
    ///
    /// <>
    fn _as_proj_string(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _as_projjson(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_source_crs(&self) { unimplemented!() }

    ///# References
    ///
    /// <>
    fn _identify(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_geoid_models_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_authorities_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_codes_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_celestial_body_list_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_crs_info_list_from_database(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
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
    ///# References
    ///
    /// <>
    fn _crs_is_derived(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_get_geodetic_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_get_horizontal_datum(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_get_sub_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_get_datum(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_get_datum_ensemble(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_get_datum_forced(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_has_point_motion_operation(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _datum_ensemble_get_member_count(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _datum_ensemble_get_accuracy(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _datum_ensemble_get_member(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _dynamic_datum_get_frame_reference_epoch(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_get_coordinate_system(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _cs_get_type(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _cs_get_axis_count(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _cs_get_axis_info(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_ellipsoid(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _ellipsoid_get_parameters(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_celestial_body_name(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _get_prime_meridian(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _prime_meridian_get_parameters(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _crs_get_coordoperation(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_get_method_info(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_is_instantiable(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_has_ballpark_transformation(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_requires_per_coordinate_input_time(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_get_param_count(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_get_param_index(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_get_param(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_get_grid_used_count(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_get_grid_used(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_get_accuracy(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_get_towgs84_values(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordoperation_create_inverse(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _concatoperation_get_step_count(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _concatoperation_get_step(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordinate_metadata_create(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _coordinate_metadata_get_epoch(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_cs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_cartesian_2d_cs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_ellipsoidal_2d_cs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_ellipsoidal_3d_cs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _query_geodetic_crs_from_datum(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_geographic_crs(&self) { unimplemented!() }
    ///# References
    ///
    /// <>
    fn _create_geographic_crs_from_datum(&self) { unimplemented!() }
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
    fn _crs_create_bound_crs_to_wgs84(&self) { unimplemented!() }
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
/// # ISO-19111
impl Pj<'_> {
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_type>
    pub fn get_type(&self) -> miette::Result<PjType> {
        let result = unsafe { proj_sys::proj_get_type(self.ptr) };
        PjType::try_from(result).into_diagnostic()
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_deprecated>
    pub fn is_deprecated(&self) -> bool { unsafe { proj_sys::proj_is_deprecated(self.ptr) != 0 } }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_equivalent_to>
    pub fn is_equivalent_to(&self, other: &Pj, criterion: PjComparisonCriterion) -> bool {
        unsafe { proj_sys::proj_is_equivalent_to(self.ptr, other.ptr, criterion.into()) != 0 }
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_crs>
    pub fn is_crs(&self) -> bool { unsafe { proj_sys::proj_is_crs(self.ptr) != 0 } }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_name>
    pub fn get_name(&self) -> String {
        crate::c_char_to_string(unsafe { proj_sys::proj_get_name(self.ptr) }).unwrap_or_default()
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_id_auth_name>
    pub fn get_id_auth_name(&self, index: i32) -> Option<String> {
        crate::c_char_to_string(unsafe { proj_sys::proj_get_id_auth_name(self.ptr, index) })
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_id_code>
    pub fn get_id_code(&self, index: i32) -> Option<String> {
        crate::c_char_to_string(unsafe { proj_sys::proj_get_id_code(self.ptr, index) })
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_remarks>
    pub fn get_remarks(&self) -> String {
        crate::c_char_to_string(unsafe { proj_sys::proj_get_remarks(self.ptr) }).unwrap_or_default()
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_domain_count>
    pub fn get_domain_count(&self) -> miette::Result<u32> {
        let count = unsafe { proj_sys::proj_get_domain_count(self.ptr) };
        if count == 0 {
            miette::bail!("get_domain_count error.")
        };
        Ok(count as u32)
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_scope>
    pub fn get_scope(&self) -> Option<String> {
        crate::c_char_to_string(unsafe { proj_sys::proj_get_scope(self.ptr) })
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_scope_ex>
    pub fn get_scope_ex(&self, domain_idx: i32) -> Option<String> {
        crate::c_char_to_string(unsafe { proj_sys::proj_get_scope_ex(self.ptr, domain_idx) })
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_as_wkt>
    // pub fn as_wkt(
    //     &self,
    //     wkt_type: PjWktType,
    //     multiline: Option<bool>,
    //     indentation_width: Option<usize>,
    //     output_axis: Option<bool>,
    //     strict: Option<bool>,
    //     allow_ellipsoidal_height_as_vertical_crs: Option<bool>,
    //     allow_linunit_node: Option<bool>,
    // ) -> miette::Result<String> {
    //     let mut options = crate::PjOptions::new(6);
    //     options.push_optional(multiline, "MULTILINE", PJ_OPTION_YES);
    //     options.push_optional(indentation_width, "INDENTATION_WIDTH", "4");
    //     options.push_optional(output_axis, "OUTPUT_AXIS", "AUTO");
    //     options.push_optional(strict, "STRICT", PJ_OPTION_YES);
    //     options.push_optional(
    //         allow_ellipsoidal_height_as_vertical_crs,
    //         "ALLOW_ELLIPSOIDAL_HEIGHT_AS_VERTICAL_CRS",
    //         PJ_OPTION_NO,
    //     );
    //     options.push_optional(allow_linunit_node, "ALLOW_LINUNIT_NODE", PJ_OPTION_YES);
    //     let result = c_char_to_string(unsafe {
    //         proj_sys::proj_as_wkt(self.ctx.ptr, self.ptr, wkt_type.into(), std::ptr::null())
    //     });
    //     check_result!(self);
    //     Ok(result.expect("Error"))
    // }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_target_crs>
    pub fn get_target_crs(&self) -> Option<Pj<'_>> {
        let out_ptr = unsafe { proj_sys::proj_get_target_crs(self.ctx.ptr, self.ptr) };
        if out_ptr.is_null() {
            return None;
        }
        Some(Self {
            ptr: out_ptr,
            ctx: self.ctx,
        })
    }
}
///# References
/// <>
fn _proj_string_list_destroy() { unimplemented!() }
///# References
/// <>
fn _proj_int_list_destroy() { unimplemented!() }
///# References
/// <>
fn _proj_celestial_body_list_destroy() { unimplemented!() }
///# References
/// <>
fn _proj_get_crs_list_parameters_create() { unimplemented!() }
///# References
/// <>
fn _proj_get_crs_list_parameters_destroy() { unimplemented!() }
///# References
/// <>
fn _proj_crs_info_list_destroy() { unimplemented!() }
///# References
/// <>
fn _proj_unit_list_destroy() { unimplemented!() }
///# References
/// <>
fn _proj_insert_object_session_create() { unimplemented!() }
///# References
/// <>
fn _proj_string_destroy() { unimplemented!() }
///# References
/// <>
fn _proj_operation_factory_context_destroy() { unimplemented!() }
///# References
/// <>
fn _proj_list_get_count() { unimplemented!() }
///# References
/// <>
fn _proj_list_destroy() { unimplemented!() }

#[cfg(test)]
mod test_context {}
#[cfg(test)]
mod test_proj {
    use crate::PjArea;
    use crate::data_types::iso19111::PjComparisonCriterion;

    #[test]
    fn test_get_type() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let t = pj.get_type()?;
        println!("{t:?}");
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
        let equivalent = pj1.is_equivalent_to(&pj2, PjComparisonCriterion::Equivalent);
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
    pub fn test_get_target_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_proj(crate::PjParams::CrsToCrs {
            source_crs: "EPSG:4326",
            target_crs: "EPSG:3857",
            area: &PjArea::default(),
        })?;
        let target = pj.get_target_crs().unwrap();
        assert_eq!(target.get_name(), "WGS 84 / Pseudo-Mercator");
        Ok(())
    }
    // #[test]
    // pub fn test_as_wkt() -> miette::Result<()> {
    //     let ctx = crate::new_test_ctx()?;
    //     let pj = ctx.create("EPSG:4326")?;
    //     let wkt = pj.as_wkt(
    //         crate::data_types::iso19111::PjWktType::Wkt2_2019,
    //         None,
    //         None,
    //         None,
    //         None,
    //         None,
    //         None,
    //     )?;
    //     assert_eq!(wkt, "WGS 84 / Pseudo-Mercator");
    //     Ok(())
    // }
}
#[cfg(test)]
mod test_other {}
