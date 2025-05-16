use num_enum::{IntoPrimitive, TryFromPrimitive};

use crate::create_readonly_struct;
/// # References
/// <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GUESSED_WKT_DIALECT>
#[derive(Debug, IntoPrimitive, TryFromPrimitive, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum PjGuessedWktDialect {
    Wkt2_2019 = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT2_2019,
    // Deprecated alias for PJ_GUESSED_WKT2_2019
    // Wkt2_2018 = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT2_2018,
    Wkt2_2015 = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT2_2015,
    Wkt1Gdal = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT1_GDAL,
    Wkt1Esri = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT1_ESRI,
    NotWkt = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_NOT_WKT,
}

/// # References
/// <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_CATEGORY>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjCategory {
    Ellipsoid,
    PrimeMeridian,
    Datum,
    Crs,
    CoordinateOperation,
    DatumEnsemble,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_TYPE>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u32)]
pub enum PjType {
    Unknown = proj_sys::PJ_TYPE_PJ_TYPE_UNKNOWN,
    Ellipsoid = proj_sys::PJ_TYPE_PJ_TYPE_ELLIPSOID,
    PrimeMeridian = proj_sys::PJ_TYPE_PJ_TYPE_PRIME_MERIDIAN,
    GeodeticReferenceFrame = proj_sys::PJ_TYPE_PJ_TYPE_GEODETIC_REFERENCE_FRAME,
    DynamicGeodeticReferenceFrame = proj_sys::PJ_TYPE_PJ_TYPE_DYNAMIC_GEODETIC_REFERENCE_FRAME,
    VerticalReferenceFrame = proj_sys::PJ_TYPE_PJ_TYPE_VERTICAL_REFERENCE_FRAME,
    DynamicVerticalReferenceFrame = proj_sys::PJ_TYPE_PJ_TYPE_DYNAMIC_VERTICAL_REFERENCE_FRAME,
    DatumEnsemble = proj_sys::PJ_TYPE_PJ_TYPE_DATUM_ENSEMBLE,
    Crs = proj_sys::PJ_TYPE_PJ_TYPE_CRS,
    GeodeticCrs = proj_sys::PJ_TYPE_PJ_TYPE_GEODETIC_CRS,
    GeocentricCrs = proj_sys::PJ_TYPE_PJ_TYPE_GEOCENTRIC_CRS,
    GeographicCr = proj_sys::PJ_TYPE_PJ_TYPE_GEOGRAPHIC_CRS,
    Geographic2dCrs = proj_sys::PJ_TYPE_PJ_TYPE_GEOGRAPHIC_2D_CRS,
    Geographic3dCrs = proj_sys::PJ_TYPE_PJ_TYPE_GEOGRAPHIC_3D_CRS,
    VerticalCrs = proj_sys::PJ_TYPE_PJ_TYPE_VERTICAL_CRS,
    ProjectedCrs = proj_sys::PJ_TYPE_PJ_TYPE_PROJECTED_CRS,
    CompoundCrs = proj_sys::PJ_TYPE_PJ_TYPE_COMPOUND_CRS,
    TemporalCrs = proj_sys::PJ_TYPE_PJ_TYPE_TEMPORAL_CRS,
    EngineeringCrs = proj_sys::PJ_TYPE_PJ_TYPE_ENGINEERING_CRS,
    BoundCrs = proj_sys::PJ_TYPE_PJ_TYPE_BOUND_CRS,
    OtherCrs = proj_sys::PJ_TYPE_PJ_TYPE_OTHER_CRS,
    Conversion = proj_sys::PJ_TYPE_PJ_TYPE_CONVERSION,
    Transformation = proj_sys::PJ_TYPE_PJ_TYPE_TRANSFORMATION,
    ConcatenatedOperation = proj_sys::PJ_TYPE_PJ_TYPE_CONCATENATED_OPERATION,
    OtherCoordinateOperation = proj_sys::PJ_TYPE_PJ_TYPE_OTHER_COORDINATE_OPERATION,
    TemporalDatum = proj_sys::PJ_TYPE_PJ_TYPE_TEMPORAL_DATUM,
    EngineeringDatum = proj_sys::PJ_TYPE_PJ_TYPE_ENGINEERING_DATUM,
    ParametricDatum = proj_sys::PJ_TYPE_PJ_TYPE_PARAMETRIC_DATUM,
    DerivedProjectedCrs = proj_sys::PJ_TYPE_PJ_TYPE_DERIVED_PROJECTED_CRS,
    CoordinateMetadata = proj_sys::PJ_TYPE_PJ_TYPE_COORDINATE_METADATA,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_COMPARISON_CRITERION>
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum PjComparisonCriterion {
    Strict = proj_sys::PJ_COMPARISON_CRITERION_PJ_COMP_STRICT,
    Equivalent = proj_sys::PJ_COMPARISON_CRITERION_PJ_COMP_EQUIVALENT,
    EquivalentExceptAxisOrderGeogcrs =
        proj_sys::PJ_COMPARISON_CRITERION_PJ_COMP_EQUIVALENT_EXCEPT_AXIS_ORDER_GEOGCRS,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_WKT_TYPE>
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum PjWktType {
    Wkt2_2015 = proj_sys::PJ_WKT_TYPE_PJ_WKT2_2015,
    Wkt2_2015Simplified = proj_sys::PJ_WKT_TYPE_PJ_WKT2_2015_SIMPLIFIED,
    Wkt2_2019 = proj_sys::PJ_WKT_TYPE_PJ_WKT2_2019,
    //Deprecated alias for PJ_WKT2_2019
    // Wkt2_2018=proj_sys::PJ_WKT_TYPE_PJ_WKT2_2018,
    Wkt2_2019Simplified = proj_sys::PJ_WKT_TYPE_PJ_WKT2_2019_SIMPLIFIED,
    //Deprecated alias for PJ_WKT2_2019
    // Wkt2_2018Simplified=proj_sys::PJ_WKT_TYPE_PJ_WKT2_2018_SIMPLIFIED,
    Wkt1Gdal = proj_sys::PJ_WKT_TYPE_PJ_WKT1_GDAL,
    Wkt1Esri = proj_sys::PJ_WKT_TYPE_PJ_WKT1_ESRI,
}

///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_EXTENT_USE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjCrsExtentUse {
    None,
    Both,
    Intersection,
    Smallest,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_GRID_AVAILABILITY_USE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjGridAvailabilityUse {
    UsedForSorting,
    DiscardOperationIfMissingGrid,
    Ignored,
    KnownAvailable,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PROJ_STRING_TYPE>

#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
#[repr(u32)]
pub enum PjStringType {
    Proj5 = proj_sys::PJ_PROJ_STRING_TYPE_PJ_PROJ_5,
    Proj4 = proj_sys::PJ_PROJ_STRING_TYPE_PJ_PROJ_4,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_SPATIAL_CRITERION>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjSpatialCriterion {
    StrictContainment,
    PartialIntersection,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_INTERMEDIATE_CRS_USE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjIntermediateCrsUse {
    Always,
    IfNoDirectTransformation,
    Never,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_COORDINATE_SYSTEM_TYPE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjCoordinateSystemType {
    Unknown,
    Cartesian,
    Ellipsoidal,
    Vertical,
    Spherical,
    Ordinal,
    Parametric,
    Datetimetemporal,
    Temporalcount,
    Temporalmeasure,
}
create_readonly_struct!(
    PjCrsInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_INFO>",
    {auth_name:String},
    {code:String},
    {name:String},
    {pj_type:PjType},
    {deprecated:bool},
    {bbox_valid:bool},
    {west_lon_degree:f64},
    {south_lat_degree:f64},
    {east_lon_degree:f64},
    {north_lat_degree:f64},
    {area_name:String},
    {projection_method_name:String},
    {celestial_body_name:String}
);

create_readonly_struct!(
    PjCrsListParameters,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_LIST_PARAMETERS>",
    {types:Vec<PjType>},
    {types_count:usize},
    {crs_area_of_use_contains_bbox:bool},
    {bbox_valid:bool},
    {west_lon_degree:f64},
    {south_lat_degree:f64},
    {east_lon_degree:f64},
    {north_lat_degree:f64},
    {allow_deprecated:bool},
    {celestial_body_name:Option<String>}
);

create_readonly_struct!(
    PjUnitInfo ,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_UNIT_INFO>",
    {auth_name: String},
    {code: String},
    {name: String},
    {category: String},
    {conv_factor: f64},
    {proj_short_name: String},
    {deprecated: bool}
);

create_readonly_struct!(
    PjCelestialBodyInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CELESTIAL_BODY_INFO>",
    {auth_name:String},
    {name:String}
);
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UNIT_TYPE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjUnitType {
    Angular,
    Linear,
    Scale,
    Time,
    Parametric,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_CARTESIAN_CS_2D_TYPE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjCartesianCs2dType {
    EastingNorthing,
    NorthingEasting,
    NorthPoleEastingSouthNorthingSouth,
    SouthPoleEastingNorthNorthingNorth,
    WestingSouthing,
}
///# References
///<>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjEllipsoidalCs2dType {
    LongitudeLatitude,
    LatitudeLongitude,
}

///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_ELLIPSOIDAL_CS_3D_TYPE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjEllipsoidalCs3dType {
    LongitudeLatitudeHeight,
    LatitudeLongitudeHeight,
}

create_readonly_struct!(
     PjAxisDescription ,
     "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_AXIS_DESCRIPTION>",
    {name: String},
    {abbreviation: String},
    {direction: String},
    {unit_name: String},
    {unit_conv_factor: f64},
    {unit_type: PjUnitType}
);

create_readonly_struct!(
    PjParamDescription ,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PARAM_DESCRIPTION>",
    {name: String},
    {auth_name: String},
    {code: String},
    {value: f64},
    {unit_name: String},
    {unit_conv_factor: f64},
    {unit_type: PjUnitType}
);
