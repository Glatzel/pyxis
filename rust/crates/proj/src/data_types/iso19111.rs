use std::ffi::CString;

use miette::IntoDiagnostic;
use num_enum::{IntoPrimitive, TryFromPrimitive};

use crate::create_readonly_struct;
/// # References
/// <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GUESSED_WKT_DIALECT>
#[derive(Debug, IntoPrimitive, TryFromPrimitive, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum GuessedWktDialect {
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
pub enum Category {
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
pub enum ProjType {
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
pub enum ComparisonCriterion {
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
pub enum WktType {
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
pub enum CrsExtentUse {
    None,
    Both,
    Intersection,
    Smallest,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_GRID_AVAILABILITY_USE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GridAvailabilityUse {
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
pub enum ProjStringType {
    Proj5 = proj_sys::PJ_PROJ_STRING_TYPE_PJ_PROJ_5,
    Proj4 = proj_sys::PJ_PROJ_STRING_TYPE_PJ_PROJ_4,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_SPATIAL_CRITERION>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SpatialCriterion {
    StrictContainment,
    PartialIntersection,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_INTERMEDIATE_CRS_USE>
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum IntermediateCrsUse {
    Always,
    IfNoDirectTransformation,
    Never,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_COORDINATE_SYSTEM_TYPE>
#[derive(Debug, PartialEq, IntoPrimitive, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum CoordinateSystemType {
    Unknown = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_UNKNOWN,
    Cartesian = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_CARTESIAN,
    Ellipsoidal = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_ELLIPSOIDAL,
    Vertical = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_VERTICAL,
    Spherical = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_SPHERICAL,
    Ordinal = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_ORDINAL,
    Parametric = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_PARAMETRIC,
    Datetimetemporal = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_DATETIMETEMPORAL,
    Temporalcount = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_TEMPORALCOUNT,
    Temporalmeasure = proj_sys::PJ_COORDINATE_SYSTEM_TYPE_PJ_CS_TYPE_TEMPORALMEASURE,
}
create_readonly_struct!(
    CrsInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_INFO>",
    {auth_name:String},
    {code:String},
    {name:String},
    {pj_type:ProjType},
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
    CrsListParameters,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_LIST_PARAMETERS>",
    {types:Vec<ProjType>},
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
    UnitInfo ,
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
    CelestialBodyInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CELESTIAL_BODY_INFO>",
    {auth_name:String},
    {name:String}
);
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UNIT_TYPE>
#[derive(Debug, Clone, Copy, IntoPrimitive, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum UnitType {
    Angular = proj_sys::PJ_UNIT_TYPE_PJ_UT_ANGULAR,
    Linear = proj_sys::PJ_UNIT_TYPE_PJ_UT_LINEAR,
    Scale = proj_sys::PJ_UNIT_TYPE_PJ_UT_SCALE,
    Time = proj_sys::PJ_UNIT_TYPE_PJ_UT_TIME,
    Parametric = proj_sys::PJ_UNIT_TYPE_PJ_UT_PARAMETRIC,
}
///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_CARTESIAN_CS_2D_TYPE>
#[derive(Debug, Clone, Copy, IntoPrimitive, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum CartesianCs2dType {
    EastingNorthing = proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_EASTING_NORTHING,
    NorthingEasting = proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_NORTHING_EASTING,
    NorthPoleEastingSouthNorthingSouth =
        proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_NORTH_POLE_EASTING_SOUTH_NORTHING_SOUTH,
    SouthPoleEastingNorthNorthingNorth =
        proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_SOUTH_POLE_EASTING_NORTH_NORTHING_NORTH,
    WestingSouthing = proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_WESTING_SOUTHING,
}

///# References
///
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_ELLIPSOIDAL_CS_2D_TYPE>
#[derive(Debug, Clone, Copy, IntoPrimitive, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum EllipsoidalCs2dType {
    LongitudeLatitude = proj_sys::PJ_ELLIPSOIDAL_CS_2D_TYPE_PJ_ELLPS2D_LONGITUDE_LATITUDE,
    LatitudeLongitude = proj_sys::PJ_ELLIPSOIDAL_CS_2D_TYPE_PJ_ELLPS2D_LATITUDE_LONGITUDE,
}

///# References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_ELLIPSOIDAL_CS_3D_TYPE>
#[derive(Debug, Clone, Copy, IntoPrimitive, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum EllipsoidalCs3dType {
    LongitudeLatitudeHeight,
    LatitudeLongitudeHeight,
}

///# References
///
/// <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_AXIS_DESCRIPTION>
pub struct AxisDescription {
    pub(crate) name: CString,
    pub(crate) abbreviation: CString,
    pub(crate) direction: CString,
    pub(crate) unit_name: CString,
    pub(crate) unit_conv_factor: f64,
    pub(crate) unit_type: UnitType,
}
impl AxisDescription {
    pub fn new(
        name: Option<&str>,
        abbreviation: Option<&str>,
        direction: AxisDirection,
        unit_name: Option<&str>,
        unit_conv_factor: f64,
        unit_type: UnitType,
    ) -> Self {
        Self {
            name: CString::new(name.unwrap_or("")).expect("Error creating CString"),
            abbreviation: CString::new(abbreviation.unwrap_or("")).expect("Error creating CString"),
            direction: direction.into(),
            unit_name: CString::new(unit_name.unwrap_or("")).expect("Error creating CString"),
            unit_conv_factor: unit_conv_factor,
            unit_type: unit_type,
        }
    }
}
create_readonly_struct!(
    ParamDescription ,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PARAM_DESCRIPTION>",
    {name: String},
    {auth_name: String},
    {code: String},
    {value: f64},
    {unit_name: String},
    {unit_conv_factor: f64},
    {unit_type: UnitType}
);

//internal
/// # References
///
/// <https://github.com/OSGeo/PROJ/blob/master/src/iso19111/static.cpp>
pub enum AxisDirection {
    North,
    NorthNorthEast,
    NorthEast,
    EastNorthEast,
    East,
    EastSouthEast,
    SouthEast,
    SouthSouthEast,
    South,
    SouthSouthWest,
    SouthWest,
    WestSouthWest,
    West,
    WestNorthWest,
    NorthWest,
    NorthNorthWest,
    Up,
    Down,
    GeocentricX,
    GeocentricY,
    GeocentricZ,
    ColumnPositive,
    ColumnNegative,
    RowPositive,
    RowNegative,
    DisplayRight,
    DisplayLeft,
    DisplayUp,
    DisplayDown,
    Forward,
    Aft,
    Port,
    Starboard,
    Clockwise,
    CounterClockwise,
    Towards,
    AwayFrom,
    Future,
    Past,
    Unspecified,
}
impl From<AxisDirection> for CString {
    fn from(value: AxisDirection) -> Self {
        CString::new(match value {
            AxisDirection::North => "north",
            AxisDirection::NorthNorthEast => "northNorthEast",
            AxisDirection::NorthEast => "northEast",
            AxisDirection::EastNorthEast => "eastNorthEast",
            AxisDirection::East => "east",
            AxisDirection::EastSouthEast => "eastSouthEast",
            AxisDirection::SouthEast => "southEast",
            AxisDirection::SouthSouthEast => "southSouthEast",
            AxisDirection::South => "south",
            AxisDirection::SouthSouthWest => "southSouthWest",
            AxisDirection::SouthWest => "southWest",
            AxisDirection::WestSouthWest => "westSouthWest",
            AxisDirection::West => "west",
            AxisDirection::WestNorthWest => "westNorthWest",
            AxisDirection::NorthWest => "northWest",
            AxisDirection::NorthNorthWest => "northNorthWest",
            AxisDirection::Up => "up",
            AxisDirection::Down => "down",
            AxisDirection::GeocentricX => "geocentricX",
            AxisDirection::GeocentricY => "geocentricY",
            AxisDirection::GeocentricZ => "geocentricZ",
            AxisDirection::ColumnPositive => "columnPositive",
            AxisDirection::ColumnNegative => "columnNegative",
            AxisDirection::RowPositive => "rowPositive",
            AxisDirection::RowNegative => "rowNegative",
            AxisDirection::DisplayRight => "displayRight",
            AxisDirection::DisplayLeft => "displayLeft",
            AxisDirection::DisplayUp => "displayUp",
            AxisDirection::DisplayDown => "displayDown",
            AxisDirection::Forward => "forward",
            AxisDirection::Aft => "aft",
            AxisDirection::Port => "port",
            AxisDirection::Starboard => "starboard",
            AxisDirection::Clockwise => "clockwise",
            AxisDirection::CounterClockwise => "counterClockwise",
            AxisDirection::Towards => "towards",
            AxisDirection::AwayFrom => "awayFrom",
            AxisDirection::Future => "future",
            AxisDirection::Past => "past",
            AxisDirection::Unspecified => "unspecified",
        })
        .expect("Error creating CString")
    }
}
create_readonly_struct!(
    EllipsoidParameters,
    "",
   {semi_major_metre: f64},
   {semi_minor_metre: f64},
   {is_semi_minor_computed :bool},
   {inv_flattening :f64}
);
create_readonly_struct!(
    PrimeMeridianParameters,
    "",
   {longitude: f64},
   {unit_conv_factor : f64},
   {unit_name :String}
);
create_readonly_struct!(
CoordOperationMethodInfo, "",
{method_name: String},
{method_auth_name : String},
{method_code :String}
);
create_readonly_struct!(
CoordOperationParam, "",
{name : String},
{auth_name  : String},
{code  :String},
{value   :f64},
{value_string   :String},
{unit_conv_factor   :f64},
{unit_name   :String},
{unit_auth_name   :String},
{unit_code   :String},
{unit_category   :UnitCategory}
);
#[derive(Debug)]
pub enum UnitCategory {
    Unknown,
    None,
    Linear,
    LinearPerTime,
    Angular,
    AngularPerTime,
    Scale,
    ScalePerTime,
    Time,
    Parametric,
    ParametricPerTime,
}
impl TryFrom<CString> for UnitCategory {
    type Error = miette::Report;

    fn try_from(value: CString) -> Result<Self, Self::Error> {
        Ok(match value.to_str().into_diagnostic()? {
            "unknown" => Self::Unknown,
            "none" => Self::None,
            "linear" => Self::Linear,
            "linear_per_time" => Self::LinearPerTime,
            "angular" => Self::Angular,
            "angular_per_time" => Self::AngularPerTime,
            "scale" => Self::Scale,
            "scale_per_time" => Self::ScalePerTime,
            "time" => Self::Time,
            "parametric" => Self::Parametric,
            "parametric_per_time" => Self::ParametricPerTime,
            _ => miette::bail!("Unknown"),
        })
    }
}
create_readonly_struct!(
CoordOperationGridUsed, "",
{short_name   : String},
{full_name   :String},
{package_name    :String},
{url    :String},
{direct_download    :bool},
{open_license    :bool},
{available    :bool}
);
