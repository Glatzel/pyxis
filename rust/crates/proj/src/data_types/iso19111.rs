use crate::create_readonly_struct;
/// # References
/// <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GUESSED_WKT_DIALECT>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjGuessedWktDialect {
    Wkt2_2019,
    Wkt2_2018,
    Wkt2_2015,
    Wkt1Gdal,
    Wkt1Esri,
    NotWkt,
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
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjType {
    Unknown,
    Ellipsoid,
    PrimeMeridian,
    GeodeticReferenceFrame,
    DynamicGeodeticReferenceFrame,
    VerticalReferenceFrame,
    DynamicVerticalReferenceFrame,
    DatumEnsemble,
    Crs,
    GeodeticCrs,
    GeocentricCrs,
    GeographicCr,
    Geographic2dCrs,
    Geographic3dCrs,
    VerticalCrs,
    ProjectedCrs,
    CompoundCrs,
    TemporalCrs,
    EngineeringCrs,
    BoundCrs,
    OtherCrs,
    Conversion,
    Transformation,
    ConcatenatedOperation,
    OtherCoordinateOperation,
    TemporalDatum,
    EngineeringDatum,
    ParametricDatum,
    DerivedProjectedCrs,
    CoordinateMetadata,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_COMPARISON_CRITERION>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjComparisonCriterion {
    Strict,
    Equivalent,
    EquivalentExceptAxisOrderGeogcrs,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_WKT_TYPE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjWktType {
    Wkt2_2015,
    Wkt2_2015Simplified,
    Wkt2_2019,
    Wkt2_2018,
    Wkt2_2019Simplified,
    Wkt2_2018Simplified,
    Wkt1Gdal,
    Wkt1Esri,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_EXTENT_USE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ProjCrsExtentUse {
    None,
    Both,
    Intersection,
    Smallest,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_GRID_AVAILABILITY_USE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ProjGridAvailabilityUse {
    UsedForSorting,
    DiscardOperationIfMissingGrid,
    Ignored,
    KnownAvailable,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PROJ_STRING_TYPE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PjProjStringType {
    Proj5,
    Proj4,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_SPATIAL_CRITERION>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ProjSpatialCriterion {
    StrictContainment,
    PartialIntersection,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_INTERMEDIATE_CRS_USE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ProjIntermediateCrsUse {
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
    ProjCrsInfo,
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
    ProjCrsListParameters,
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
    ProjUnitInfo ,
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
    ProjCelestialBodyInfo,
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
