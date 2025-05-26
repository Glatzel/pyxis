use std::ffi::CString;

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
#[derive(Debug, IntoPrimitive, TryFromPrimitive)]
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
        name: AxisName,
        abbreviation: AxisAbbreviation,
        direction: AxisDirection,
        unit_name: UnitName,
        unit_conv_factor: f64,
        unit_type: UnitType,
    ) -> Self {
        Self {
            name: name.into(),
            abbreviation: abbreviation.into(),
            direction: direction.into(),
            unit_name: unit_name.into(),
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

//implicit
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AxisName {
    Longitude,
    Latitude,
    Easting,
    Northing,
    Westing,
    Southing,
    EllipsoidalHeight,
    GeocentricX,
    GeocentricY,
    GeocentricZ,
}
impl From<AxisName> for CString {
    fn from(value: AxisName) -> Self {
        CString::new(match value {
            AxisName::Longitude => "Longitude",
            AxisName::Latitude => "Latitude",
            AxisName::Easting => "Easting",
            AxisName::Northing => "Northing",
            AxisName::Westing => "Westing",
            AxisName::Southing => "Southing",
            AxisName::EllipsoidalHeight => "Ellipsoidal height",
            AxisName::GeocentricX => "Geocentric X",
            AxisName::GeocentricY => "Geocentric Y",
            AxisName::GeocentricZ => "Geocentric Z",
        })
        .expect("Error creating CString")
    }
}
pub enum AxisAbbreviation {
    Lon,
    Lat,
    E,
    N,
    H,
    X,
    Y,
    Z,
}
impl From<AxisAbbreviation> for CString {
    fn from(value: AxisAbbreviation) -> Self {
        CString::new(match value {
            AxisAbbreviation::Lon => "lon",
            AxisAbbreviation::Lat => "lat",
            AxisAbbreviation::E => "E",
            AxisAbbreviation::N => "N",
            AxisAbbreviation::H => "h",
            AxisAbbreviation::X => "X",
            AxisAbbreviation::Y => "Y",
            AxisAbbreviation::Z => "Z",
        })
        .expect("Error creating CString")
    }
}
#[derive()]
pub enum AxisDirection {
    North = 0,
    NorthNorthEast = 1,
    NorthEast = 2,
    EastNorthEast = 3,
    East = 4,
    EastSouthEast = 5,
    SouthEast = 6,
    SouthSouthEast = 7,
    South = 8,
    SouthSouthWest = 9,
    SouthWest = 10,
    WestSouthWest = 11,
    West = 12,
    WestNorthWest = 13,
    NorthWest = 14,
    NorthNorthWest = 15,
    Up = 16,
    Down = 17,
    GeocentricX = 18,
    GeocentricY = 19,
    GeocentricZ = 20,
    ColumnPositive = 21,
    ColumnNegative = 22,
    RowPositive = 23,
    RowNegative = 24,
    DisplayRight = 25,
    DisplayLeft = 26,
    DisplayUp = 27,
    DisplayDown = 28,
    Forward = 29,
    Aft = 30,
    Port = 31,
    Starboard = 32,
    Clockwise = 33,
    CounterClockwise = 34,
    Towards = 35,
    AwayFrom = 36,
    Future = 37,
    Past = 38,
    Unspecified = 39,
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
pub enum UnitName {
    None,
    ScaleUnity,
    PartsPerMillion,
    Metre,
    Foot,
    UsFoot,
    Degree,
    ArcSecond,
    Grad,
    Radian,
    Microradian,
    Second,
    Year,
    MetrePerYear,
    ArcSecondPerYear,
    PpmPerYear,
}
impl From<UnitName> for CString {
    fn from(value: UnitName) -> Self {
        CString::new(match value {
            UnitName::None => "",
            UnitName::ScaleUnity => "unity",
            UnitName::PartsPerMillion => "parts per million",
            UnitName::Metre => "metre",
            UnitName::Foot => "foot",
            UnitName::UsFoot => "US survey foot",
            UnitName::Degree => "degree",
            UnitName::ArcSecond => "arc-second",
            UnitName::Grad => "grad",
            UnitName::Radian => "radian",
            UnitName::Microradian => "microradian",
            UnitName::Second => "second",
            UnitName::Year => "year",
            UnitName::MetrePerYear => "metres per year",
            UnitName::ArcSecondPerYear => "arc-seconds per year",
            UnitName::PpmPerYear => "parts per million per year",
        })
        .expect("Error creating CString")
    }
}
