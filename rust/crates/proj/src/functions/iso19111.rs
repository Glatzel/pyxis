impl crate::PjContext {
    pub fn context_set_autoclose_database(&self) {
        unimplemented!()
    }
    pub fn context_set_database_path(&self) {
        unimplemented!()
    }
    pub fn context_get_database_path(&self) {
        unimplemented!()
    }
    pub fn context_get_database_metadata(&self) {
        unimplemented!()
    }
    pub fn context_get_database_structure(&self) {
        unimplemented!()
    }
    pub fn context_guess_wkt_dialect(&self) {
        unimplemented!()
    }
    pub fn create_from_wkt(&self) {
        unimplemented!()
    }
    pub fn create_from_database(&self) {
        unimplemented!()
    }
    pub fn uom_get_info_from_database(&self) {
        unimplemented!()
    }
    pub fn grid_get_info_from_database(&self) {
        unimplemented!()
    }
    pub fn clone(&self) {
        unimplemented!()
    }
    pub fn create_from_name(&self) {
        unimplemented!()
    }
    pub fn get_non_deprecated(&self) {
        unimplemented!()
    }
    pub fn is_equivalent_to_with_ctx(&self) {
        unimplemented!()
    }
    pub fn get_area_of_use(&self) {
        unimplemented!()
    }
    pub fn get_area_of_use_ex(&self) {
        unimplemented!()
    }
    pub fn as_wkt(&self) {
        unimplemented!()
    }
    pub fn as_proj_string(&self) {
        unimplemented!()
    }
    pub fn as_projjson(&self) {
        unimplemented!()
    }
    pub fn get_source_crs(&self) {
        unimplemented!()
    }
    pub fn get_target_crs(&self) {
        unimplemented!()
    }
    pub fn identify(&self) {
        unimplemented!()
    }
    pub fn get_geoid_models_from_database(&self) {
        unimplemented!()
    }
    pub fn get_authorities_from_database(&self) {
        unimplemented!()
    }
    pub fn get_codes_from_database(&self) {
        unimplemented!()
    }
    pub fn get_celestial_body_list_from_database(&self) {
        unimplemented!()
    }
    pub fn get_crs_info_list_from_database(&self) {
        unimplemented!()
    }
    pub fn get_units_from_database(&self) {
        unimplemented!()
    }
    pub fn insert_object_session_create(&self) {
        unimplemented!()
    }
    pub fn insert_object_session_destroy(&self) {
        unimplemented!()
    }
    pub fn get_insert_statements(&self) {
        unimplemented!()
    }
    pub fn suggests_code_for(&self) {
        unimplemented!()
    }
    pub fn create_operation_factory_context(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_desired_accuracy(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_area_of_interest(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_area_of_interest_name(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_crs_extent_use(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_spatial_criterion(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_grid_availability_use(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_use_proj_alternative_grid_names(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_allow_use_intermediate_crs(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_allowed_intermediate_crs(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_discard_superseded(&self) {
        unimplemented!()
    }
    pub fn operation_factory_context_set_allow_ballpark_transformations(&self) {
        unimplemented!()
    }
    pub fn create_operations(&self) {
        unimplemented!()
    }
    pub fn list_get(&self) {
        unimplemented!()
    }
    pub fn get_suggested_operation(&self) {
        unimplemented!()
    }
    pub fn crs_is_derived(&self) {
        unimplemented!()
    }
    pub fn crs_get_geodetic_crs(&self) {
        unimplemented!()
    }
    pub fn crs_get_horizontal_datum(&self) {
        unimplemented!()
    }
    pub fn crs_get_sub_crs(&self) {
        unimplemented!()
    }
    pub fn crs_get_datum(&self) {
        unimplemented!()
    }
    pub fn crs_get_datum_ensemble(&self) {
        unimplemented!()
    }
    pub fn crs_get_datum_forced(&self) {
        unimplemented!()
    }
    pub fn crs_has_point_motion_operation(&self) {
        unimplemented!()
    }
    pub fn datum_ensemble_get_member_count(&self) {
        unimplemented!()
    }
    pub fn datum_ensemble_get_accuracy(&self) {
        unimplemented!()
    }
    pub fn datum_ensemble_get_member(&self) {
        unimplemented!()
    }
    pub fn dynamic_datum_get_frame_reference_epoch(&self) {
        unimplemented!()
    }
    pub fn crs_get_coordinate_system(&self) {
        unimplemented!()
    }
    pub fn cs_get_type(&self) {
        unimplemented!()
    }
    pub fn cs_get_axis_count(&self) {
        unimplemented!()
    }
    pub fn cs_get_axis_info(&self) {
        unimplemented!()
    }
    pub fn get_ellipsoid(&self) {
        unimplemented!()
    }
    pub fn ellipsoid_get_parameters(&self) {
        unimplemented!()
    }
    pub fn get_celestial_body_name(&self) {
        unimplemented!()
    }
    pub fn get_prime_meridian(&self) {
        unimplemented!()
    }
    pub fn prime_meridian_get_parameters(&self) {
        unimplemented!()
    }
    pub fn crs_get_coordoperation(&self) {
        unimplemented!()
    }
    pub fn coordoperation_get_method_info(&self) {
        unimplemented!()
    }
    pub fn coordoperation_is_instantiable(&self) {
        unimplemented!()
    }
    pub fn coordoperation_has_ballpark_transformation(&self) {
        unimplemented!()
    }
    pub fn coordoperation_requires_per_coordinate_input_time(&self) {
        unimplemented!()
    }
    pub fn coordoperation_get_param_count(&self) {
        unimplemented!()
    }
    pub fn coordoperation_get_param_index(&self) {
        unimplemented!()
    }
    pub fn coordoperation_get_param(&self) {
        unimplemented!()
    }
    pub fn coordoperation_get_grid_used_count(&self) {
        unimplemented!()
    }
    pub fn coordoperation_get_grid_used(&self) {
        unimplemented!()
    }
    pub fn coordoperation_get_accuracy(&self) {
        unimplemented!()
    }
    pub fn coordoperation_get_towgs84_values(&self) {
        unimplemented!()
    }
    pub fn coordoperation_create_inverse(&self) {
        unimplemented!()
    }
    pub fn concatoperation_get_step_count(&self) {
        unimplemented!()
    }
    pub fn concatoperation_get_step(&self) {
        unimplemented!()
    }
    pub fn coordinate_metadata_create(&self) {
        unimplemented!()
    }
    pub fn coordinate_metadata_get_epoch(&self) {
        unimplemented!()
    }
    pub fn create_cs(&self) {
        unimplemented!()
    }
    pub fn create_cartesian_2d_cs(&self) {
        unimplemented!()
    }
    pub fn create_ellipsoidal_2d_cs(&self) {
        unimplemented!()
    }
    pub fn create_ellipsoidal_3d_cs(&self) {
        unimplemented!()
    }
    pub fn query_geodetic_crs_from_datum(&self) {
        unimplemented!()
    }
    pub fn create_geographic_crs(&self) {
        unimplemented!()
    }
    pub fn create_geographic_crs_from_datum(&self) {
        unimplemented!()
    }
    pub fn create_geocentric_crs(&self) {
        unimplemented!()
    }
    pub fn create_geocentric_crs_from_datum(&self) {
        unimplemented!()
    }
    pub fn create_derived_geographic_crs(&self) {
        unimplemented!()
    }
    pub fn is_derived_crs(&self) {
        unimplemented!()
    }
    pub fn alter_name(&self) {
        unimplemented!()
    }
    pub fn alter_id(&self) {
        unimplemented!()
    }
    pub fn crs_alter_geodetic_crs(&self) {
        unimplemented!()
    }
    pub fn crs_alter_cs_angular_unit(&self) {
        unimplemented!()
    }
    pub fn crs_alter_cs_linear_unit(&self) {
        unimplemented!()
    }
    pub fn crs_alter_parameters_linear_unit(&self) {
        unimplemented!()
    }
    pub fn crs_promote_to_3d(&self) {
        unimplemented!()
    }
    pub fn crs_create_projected_3d_crs_from_2d(&self) {
        unimplemented!()
    }
    pub fn crs_demote_to_2d(&self) {
        unimplemented!()
    }
    pub fn create_engineering_crs(&self) {
        unimplemented!()
    }
    pub fn create_vertical_crs(&self) {
        unimplemented!()
    }
    pub fn create_vertical_crs_ex(&self) {
        unimplemented!()
    }
    pub fn create_compound_crs(&self) {
        unimplemented!()
    }
    pub fn create_conversion(&self) {
        unimplemented!()
    }
    pub fn create_transformation(&self) {
        unimplemented!()
    }
    pub fn convert_conversion_to_other_method(&self) {
        unimplemented!()
    }
    pub fn create_projected_crs(&self) {
        unimplemented!()
    }
    pub fn crs_create_bound_crs(&self) {
        unimplemented!()
    }
    pub fn crs_create_bound_crs_to_wgs84(&self) {
        unimplemented!()
    }
    pub fn crs_create_bound_vertical_crs(&self) {
        unimplemented!()
    }
    pub fn create_conversion_utm(&self) {
        unimplemented!()
    }
    pub fn create_conversion_transverse_mercator(&self) {
        unimplemented!()
    }
    pub fn create_conversion_gauss_schreiber_transverse_mercator(&self) {
        unimplemented!()
    }
    pub fn create_conversion_transverse_mercator_south_oriented(&self) {
        unimplemented!()
    }
    pub fn create_conversion_two_point_equidistant(&self) {
        unimplemented!()
    }
    pub fn create_conversion_tunisia_mapping_grid(&self) {
        unimplemented!()
    }
    pub fn create_conversion_tunisia_mining_grid(&self) {
        unimplemented!()
    }
    pub fn create_conversion_albers_equal_area(&self) {
        unimplemented!()
    }
    pub fn create_conversion_lambert_conic_conformal_1sp(&self) {
        unimplemented!()
    }
    pub fn create_conversion_lambert_conic_conformal_1sp_variant_b(&self) {
        unimplemented!()
    }
    pub fn create_conversion_lambert_conic_conformal_2sp(&self) {
        unimplemented!()
    }
    pub fn create_conversion_lambert_conic_conformal_2sp_michigan(&self) {
        unimplemented!()
    }
    pub fn create_conversion_lambert_conic_conformal_2sp_belgium(&self) {
        unimplemented!()
    }
    pub fn create_conversion_azimuthal_equidistant(&self) {
        unimplemented!()
    }
    pub fn create_conversion_guam_projection(&self) {
        unimplemented!()
    }
    pub fn create_conversion_bonne(&self) {
        unimplemented!()
    }
    pub fn create_conversion_lambert_cylindrical_equal_area_spherical(&self) {
        unimplemented!()
    }
    pub fn create_conversion_lambert_cylindrical_equal_area(&self) {
        unimplemented!()
    }
    pub fn create_conversion_cassini_soldner(&self) {
        unimplemented!()
    }
    pub fn create_conversion_equidistant_conic(&self) {
        unimplemented!()
    }
    pub fn create_conversion_eckert_i(&self) {
        unimplemented!()
    }
    pub fn create_conversion_eckert_ii(&self) {
        unimplemented!()
    }
    pub fn create_conversion_eckert_iii(&self) {
        unimplemented!()
    }
    pub fn create_conversion_eckert_iv(&self) {
        unimplemented!()
    }
    pub fn create_conversion_eckert_v(&self) {
        unimplemented!()
    }
    pub fn create_conversion_eckert_vi(&self) {
        unimplemented!()
    }
    pub fn create_conversion_equidistant_cylindrical(&self) {
        unimplemented!()
    }
    pub fn create_conversion_equidistant_cylindrical_spherical(&self) {
        unimplemented!()
    }
    pub fn create_conversion_gall(&self) {
        unimplemented!()
    }
    pub fn create_conversion_goode_homolosine(&self) {
        unimplemented!()
    }
    pub fn create_conversion_interrupted_goode_homolosine(&self) {
        unimplemented!()
    }
    pub fn create_conversion_geostationary_satellite_sweep_x(&self) {
        unimplemented!()
    }
    pub fn create_conversion_geostationary_satellite_sweep_y(&self) {
        unimplemented!()
    }
    pub fn create_conversion_gnomonic(&self) {
        unimplemented!()
    }
    pub fn create_conversion_hotine_oblique_mercator_variant_a(&self) {
        unimplemented!()
    }
    pub fn create_conversion_hotine_oblique_mercator_variant_b(&self) {
        unimplemented!()
    }
    pub fn create_conversion_hotine_oblique_mercator_two_point_natural_origin(&self) {
        unimplemented!()
    }
    pub fn create_conversion_laborde_oblique_mercator(&self) {
        unimplemented!()
    }
    pub fn create_conversion_international_map_world_polyconic(&self) {
        unimplemented!()
    }
    pub fn create_conversion_krovak_north_oriented(&self) {
        unimplemented!()
    }
    pub fn create_conversion_krovak(&self) {
        unimplemented!()
    }
    pub fn create_conversion_lambert_azimuthal_equal_area(&self) {
        unimplemented!()
    }
    pub fn create_conversion_miller_cylindrical(&self) {
        unimplemented!()
    }
    pub fn create_conversion_mercator_variant_a(&self) {
        unimplemented!()
    }
    pub fn create_conversion_mercator_variant_b(&self) {
        unimplemented!()
    }
    pub fn create_conversion_popular_visualisation_pseudo_mercator(&self) {
        unimplemented!()
    }
    pub fn create_conversion_mollweide(&self) {
        unimplemented!()
    }
    pub fn create_conversion_new_zealand_mapping_grid(&self) {
        unimplemented!()
    }
    pub fn create_conversion_oblique_stereographic(&self) {
        unimplemented!()
    }
    pub fn create_conversion_orthographic(&self) {
        unimplemented!()
    }
    pub fn create_conversion_local_orthographic(&self) {
        unimplemented!()
    }
    pub fn create_conversion_american_polyconic(&self) {
        unimplemented!()
    }
    pub fn create_conversion_polar_stereographic_variant_a(&self) {
        unimplemented!()
    }
    pub fn create_conversion_polar_stereographic_variant_b(&self) {
        unimplemented!()
    }
    pub fn create_conversion_robinson(&self) {
        unimplemented!()
    }
    pub fn create_conversion_sinusoidal(&self) {
        unimplemented!()
    }
    pub fn create_conversion_stereographic(&self) {
        unimplemented!()
    }
    pub fn create_conversion_wagner_i(&self) {
        unimplemented!()
    }
    pub fn create_conversion_wagner_ii(&self) {
        unimplemented!()
    }
    pub fn create_conversion_wagner_iii(&self) {
        unimplemented!()
    }
    pub fn create_conversion_wagner_iv(&self) {
        unimplemented!()
    }
    pub fn create_conversion_wagner_v(&self) {
        unimplemented!()
    }
    pub fn create_conversion_wagner_vi(&self) {
        unimplemented!()
    }
    pub fn create_conversion_wagner_vii(&self) {
        unimplemented!()
    }
    pub fn create_conversion_quadrilateralized_spherical_cube(&self) {
        unimplemented!()
    }
    pub fn create_conversion_spherical_cross_track_height(&self) {
        unimplemented!()
    }
    pub fn create_conversion_equal_earth(&self) {
        unimplemented!()
    }
    pub fn create_conversion_vertical_perspective(&self) {
        unimplemented!()
    }
    pub fn create_conversion_pole_rotation_grib_convention(&self) {
        unimplemented!()
    }
    pub fn create_conversion_pole_rotation_netcdf_cf_convention(&self) {
        unimplemented!()
    }
}
impl crate::Pj {
    pub fn get_type(&self) {
        unimplemented!()
    }
    pub fn is_equivalent_to(&self) {
        unimplemented!()
    }
    pub fn is_crs(&self) {
        unimplemented!()
    }
    pub fn get_name(&self) {
        unimplemented!()
    }
    pub fn get_id_auth_name(&self) {
        unimplemented!()
    }
    pub fn get_id_code(&self) {
        unimplemented!()
    }
    pub fn get_remarks(&self) {
        unimplemented!()
    }
    pub fn get_domain_count(&self) {
        unimplemented!()
    }
    pub fn get_scope(&self) {
        unimplemented!()
    }
    pub fn get_scope_ex(&self) {
        unimplemented!()
    }
}
pub fn proj_string_list_destroy() {
    unimplemented!()
}
pub fn proj_int_list_destroy() {
    unimplemented!()
}
pub fn proj_celestial_body_list_destroy() {
    unimplemented!()
}
pub fn proj_get_crs_list_parameters_create() {
    unimplemented!()
}
pub fn proj_get_crs_list_parameters_destroy() {
    unimplemented!()
}
pub fn proj_crs_info_list_destroy() {
    unimplemented!()
}
pub fn proj_unit_list_destroy() {
    unimplemented!()
}
pub fn proj_insert_object_session_create() {
    unimplemented!()
}
pub fn proj_string_destroy() {
    unimplemented!()
}
pub fn proj_operation_factory_context_destroy() {
    unimplemented!()
}
pub fn proj_list_get_count() {
    unimplemented!()
}
pub fn proj_list_destroy() {
    unimplemented!()
}
