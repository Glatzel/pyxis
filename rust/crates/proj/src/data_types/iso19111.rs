extern crate alloc;
use alloc::sync::Arc;
use core::fmt::Display;

use derive_getters::Getters;
use num_enum::TryFromPrimitive;
use strum::{AsRefStr, EnumString};

use crate::data_types::ProjError;
use crate::data_types::transformation::ContextPtr;
use crate::{OwnedCStrings, Proj, check_result};
///Guessed WKT "dialect".
///
/// # Reference
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GUESSED_WKT_DIALECT>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum GuessedWktDialect {
    ///<https://proj.org/en/stable/development/reference/cpp/cpp_general.html#general_doc_1WKT2_2019>
    Wkt2_2019 = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT2_2019,
    // Deprecated alias for PJ_GUESSED_WKT2_2019
    // Wkt2_2018 = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT2_2018,
    ///<https://proj.org/en/stable/development/reference/cpp/cpp_general.html#general_doc_1WKT2_2015>
    Wkt2_2015 = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT2_2015,
    ///<https://proj.org/en/stable/development/reference/cpp/cpp_general.html#general_doc_1WKT1>
    Wkt1Gdal = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT1_GDAL,
    ///ESRI variant of WKT1
    Wkt1Esri = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_WKT1_ESRI,
    ///Not WKT / unrecognized
    NotWkt = proj_sys::PJ_GUESSED_WKT_DIALECT_PJ_GUESSED_NOT_WKT,
}
///Object category.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_CATEGORY>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum Category {
    Ellipsoid = proj_sys::PJ_CATEGORY_PJ_CATEGORY_ELLIPSOID,
    PrimeMeridian = proj_sys::PJ_CATEGORY_PJ_CATEGORY_PRIME_MERIDIAN,
    Datum = proj_sys::PJ_CATEGORY_PJ_CATEGORY_DATUM,
    Crs = proj_sys::PJ_CATEGORY_PJ_CATEGORY_CRS,
    CoordinateOperation = proj_sys::PJ_CATEGORY_PJ_CATEGORY_COORDINATE_OPERATION,
    DatumEnsemble = proj_sys::PJ_CATEGORY_PJ_CATEGORY_DATUM_ENSEMBLE,
}
///Object type.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_TYPE>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(test, derive(strum::EnumIter))]
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
///Comparison criterion.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_COMPARISON_CRITERION>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum ComparisonCriterion {
    ///All properties are identical.
    Strict = proj_sys::PJ_COMPARISON_CRITERION_PJ_COMP_STRICT,
    ///The objects are equivalent for the purpose of coordinate operations.
    /// They can differ by the name of their objects, identifiers, other
    /// metadata. Parameters may be expressed in different units, provided that
    /// the value is (with some tolerance) the same once expressed in a common
    /// unit.
    Equivalent = proj_sys::PJ_COMPARISON_CRITERION_PJ_COMP_EQUIVALENT,
    ///Same as EQUIVALENT, relaxed with an exception that the axis order of the
    /// base CRS of a DerivedCRS/ProjectedCRS or the axis order of a
    /// GeographicCRS is ignored. Only to be used with
    /// DerivedCRS/ProjectedCRS/GeographicCRS
    EquivalentExceptAxisOrderGeogcrs =
        proj_sys::PJ_COMPARISON_CRITERION_PJ_COMP_EQUIVALENT_EXCEPT_AXIS_ORDER_GEOGCRS,
}
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_WKT_TYPE>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum WktType {
    ///<https://proj.org/en/stable/development/reference/cpp/io.html#classosgeo_1_1proj_1_1io_1_1WKTFormatter_1ae94f4401c1eeae3808dce1aaa8d25f42acbbf33e2fa1d0e7754df8c2ab40bf7a2>
    Wkt2_2015 = proj_sys::PJ_WKT_TYPE_PJ_WKT2_2015,
    ///<https://proj.org/en/stable/development/reference/cpp/io.html#classosgeo_1_1proj_1_1io_1_1WKTFormatter_1ae94f4401c1eeae3808dce1aaa8d25f42a8a4e61323a3ab9204ff3ac3cd8b23c39>
    Wkt2_2015Simplified = proj_sys::PJ_WKT_TYPE_PJ_WKT2_2015_SIMPLIFIED,
    ///<https://proj.org/en/stable/development/reference/cpp/io.html#classosgeo_1_1proj_1_1io_1_1WKTFormatter_1ae94f4401c1eeae3808dce1aaa8d25f42ac634e196cf84127855e2ff4569674d0d>
    Wkt2_2019 = proj_sys::PJ_WKT_TYPE_PJ_WKT2_2019,
    //Deprecated alias for PJ_WKT2_2019
    // Wkt2_2018=proj_sys::PJ_WKT_TYPE_PJ_WKT2_2018,
    ///<https://proj.org/en/stable/development/reference/cpp/io.html#classosgeo_1_1proj_1_1io_1_1WKTFormatter_1ae94f4401c1eeae3808dce1aaa8d25f42a1a237b13d56f5b895c4e3abf9749783e>
    Wkt2_2019Simplified = proj_sys::PJ_WKT_TYPE_PJ_WKT2_2019_SIMPLIFIED,
    //Deprecated alias for PJ_WKT2_2019
    // Wkt2_2018Simplified=proj_sys::PJ_WKT_TYPE_PJ_WKT2_2018_SIMPLIFIED,
    ///<https://proj.org/en/stable/development/reference/cpp/io.html#classosgeo_1_1proj_1_1io_1_1WKTFormatter_1ae94f4401c1eeae3808dce1aaa8d25f42a85c43e48faba72b30e6501b41536afe5>
    Wkt1Gdal = proj_sys::PJ_WKT_TYPE_PJ_WKT1_GDAL,
    ///<https://proj.org/en/stable/development/reference/cpp/io.html#classosgeo_1_1proj_1_1io_1_1WKTFormatter_1ae94f4401c1eeae3808dce1aaa8d25f42a8da08577d1e0b736b2259c71c40f0e38>
    Wkt1Esri = proj_sys::PJ_WKT_TYPE_PJ_WKT1_ESRI,
}
/// Specify how source and target CRS extent should be used to restrict
/// candidate operations (only taken into account if no explicit area of
/// interest is specified. # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_EXTENT_USE>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[repr(u32)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CrsExtentUse {
    ///Ignore CRS extent
    None = proj_sys::PROJ_CRS_EXTENT_USE_PJ_CRS_EXTENT_NONE,
    ///Test coordinate operation extent against both CRS extent.
    Both = proj_sys::PROJ_CRS_EXTENT_USE_PJ_CRS_EXTENT_BOTH,
    ///Test coordinate operation extent against the intersection of both CRS
    /// extent.
    Intersection = proj_sys::PROJ_CRS_EXTENT_USE_PJ_CRS_EXTENT_INTERSECTION,
    ///Test coordinate operation against the smallest of both CRS extent.
    Smallest = proj_sys::PROJ_CRS_EXTENT_USE_PJ_CRS_EXTENT_SMALLEST,
}
///Describe how grid availability is used.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_GRID_AVAILABILITY_USE>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[repr(u32)]
pub enum GridAvailabilityUse {
    ///Grid availability is only used for sorting results. Operations where
    /// some grids are missing will be sorted last.
    UsedForSorting = proj_sys::PROJ_GRID_AVAILABILITY_USE_PROJ_GRID_AVAILABILITY_USED_FOR_SORTING,
    ///Completely discard an operation if a required grid is missing.
    DiscardOperationIfMissingGrid = proj_sys::PROJ_GRID_AVAILABILITY_USE_PROJ_GRID_AVAILABILITY_DISCARD_OPERATION_IF_MISSING_GRID,
    ///Ignore grid availability at all. Results will be presented as if all
    /// grids were available.
    Ignored = proj_sys::PROJ_GRID_AVAILABILITY_USE_PROJ_GRID_AVAILABILITY_IGNORED,
    ///Results will be presented as if grids known to PROJ (that is registered
    /// in the grid_alternatives table of its database) were available. Used
    /// typically when networking is enabled.
    KnownAvailable = proj_sys::PROJ_GRID_AVAILABILITY_USE_PROJ_GRID_AVAILABILITY_KNOWN_AVAILABLE,
}
///PROJ string version.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PROJ_STRING_TYPE>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[repr(u32)]
pub enum ProjStringType {
    ///cf [osgeo::proj::io::PROJStringFormatter::Convention::PROJ_5](https://proj.org/en/stable/development/reference/cpp/io.html#classosgeo_1_1proj_1_1io_1_1PROJStringFormatter_1a797997db6984aa2bad279abb0010ff13a475fc81228e34a4715d2d28f4d7f2851)
    Proj5 = proj_sys::PJ_PROJ_STRING_TYPE_PJ_PROJ_5,
    ///cf [osgeo::proj::io::PROJStringFormatter::Convention::PROJ_4](https://proj.org/en/stable/development/reference/cpp/io.html#classosgeo_1_1proj_1_1io_1_1PROJStringFormatter_1a797997db6984aa2bad279abb0010ff13ae3bec874928ae377030a07a550bdc7eb)
    Proj4 = proj_sys::PJ_PROJ_STRING_TYPE_PJ_PROJ_4,
}
///Spatial criterion to restrict candidate operations.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_SPATIAL_CRITERION>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum SpatialCriterion {
    ///The area of validity of transforms should strictly contain the are of
    /// interest.
    StrictContainment = proj_sys::PROJ_SPATIAL_CRITERION_PROJ_SPATIAL_CRITERION_STRICT_CONTAINMENT,
    ///The area of validity of transforms should at least intersect the area of
    /// interest.
    PartialIntersection =
        proj_sys::PROJ_SPATIAL_CRITERION_PROJ_SPATIAL_CRITERION_PARTIAL_INTERSECTION,
}
///Describe if and how intermediate CRS should be used
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_INTERMEDIATE_CRS_USE>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum IntermediateCrsUse {
    ///Always search for intermediate CRS.
    Always = proj_sys::PROJ_INTERMEDIATE_CRS_USE_PROJ_INTERMEDIATE_CRS_USE_ALWAYS,
    ///Only attempt looking for intermediate CRS if there is no direct
    /// transformation available.
    IfNoDirectTransformation =
        proj_sys::PROJ_INTERMEDIATE_CRS_USE_PROJ_INTERMEDIATE_CRS_USE_IF_NO_DIRECT_TRANSFORMATION,
    Never = proj_sys::PROJ_INTERMEDIATE_CRS_USE_PROJ_INTERMEDIATE_CRS_USE_NEVER,
}
///Type of coordinate system.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_COORDINATE_SYSTEM_TYPE>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
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
/// Structure given overall description of a CRS.
///
/// This structure may grow over time, and should not be directly allocated by
/// client code.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_INFO>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct CrsInfo {
    /// Authority name.
    auth_name: String,
    /// Object code.
    code: String,
    /// Object name.
    name: String,
    /// Object type.
    pj_type: ProjType,
    /// Whether the object is deprecated
    deprecated: bool,
    /// Whereas the west_lon_degree, south_lat_degree, east_lon_degree and
    bbox_valid: bool,
    /// Western-most longitude of the area of use, in degrees.
    west_lon_degree: f64,
    /// Southern-most latitude of the area of use, in degrees.
    south_lat_degree: f64,
    /// Eastern-most longitude of the area of use, in degrees.
    east_lon_degree: f64,
    /// Northern-most latitude of the area of use, in degrees.
    north_lat_degree: f64,
    /// Name of the area of use.
    area_name: String,
    /// Name of the projection method for a projected CRS. Might be NULL even
    projection_method_name: String,
    /// Name of the celestial body of the CRS (e.g. `Earth`).
    celestial_body_name: String,
}
impl CrsInfo {
    pub fn new(
        auth_name: String,
        code: String,
        name: String,
        pj_type: ProjType,
        deprecated: bool,
        bbox_valid: bool,
        west_lon_degree: f64,
        south_lat_degree: f64,
        east_lon_degree: f64,
        north_lat_degree: f64,
        area_name: String,
        projection_method_name: String,
        celestial_body_name: String,
    ) -> Self {
        CrsInfo {
            auth_name,
            code,
            name,
            pj_type,
            deprecated,
            bbox_valid,
            west_lon_degree,
            south_lat_degree,
            east_lon_degree,
            north_lat_degree,
            area_name,
            projection_method_name,
            celestial_body_name,
        }
    }
}

///Structure describing optional parameters for proj_get_crs_list();.
///
///This structure may grow over time, and should not be directly allocated by
/// client code.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CRS_LIST_PARAMETERS>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct CrsListParameters {
    /// Array of allowed object types. Should be NULL if all types are allowed
    types: Vec<ProjType>,
    /// If TRUE and bbox_valid == TRUE, then only CRS whose area of use entirely
    /// contains the specified bounding box will be returned.
    crs_area_of_use_contains_bbox: bool,
    /// If TRUE and bbox_valid == TRUE, then only CRS whose area of use
    bbox_valid: bool,
    /// Western-most longitude of the area of use, in degrees.
    west_lon_degree: f64,
    /// Southern-most latitude of the area of use, in degrees.
    south_lat_degree: f64,
    /// Eastern-most longitude of the area of use, in degrees.
    east_lon_degree: f64,
    /// Northern-most latitude of the area of use, in degrees.
    north_lat_degree: f64,
    /// Whether deprecated objects are allowed. Default to FALSE.
    allow_deprecated: bool,
    /// Celestial body of the CRS (e.g.` Earth`). The default value, NULL,
    celestial_body_name: Option<String>,
}
impl CrsListParameters {
    pub fn new(
        types: Vec<ProjType>,
        crs_area_of_use_contains_bbox: bool,
        bbox_valid: bool,
        west_lon_degree: f64,
        south_lat_degree: f64,
        east_lon_degree: f64,
        north_lat_degree: f64,
        allow_deprecated: bool,
        celestial_body_name: Option<String>,
    ) -> Self {
        CrsListParameters {
            types,
            crs_area_of_use_contains_bbox,
            bbox_valid,
            west_lon_degree,
            south_lat_degree,
            east_lon_degree,
            north_lat_degree,
            allow_deprecated,
            celestial_body_name,
        }
    }
}

///Structure given description of a unit.
///
///This structure may grow over time, and should not be directly allocated by
/// client code.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_UNIT_INFO>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct UnitInfo {
    /// Authority name.
    auth_name: String,
    /// Object code.
    code: String,
    /// Object name. For example `metre`, `US survey foot`, etc.
    name: String,
    /// Category of the unit: one of `linear`, `linear_per_time`, `angular`,
    /// `angular_per_time`, `scale`, `scale_per_time` or `time`
    category: UnitCategory,
    /// Conversion factor to apply to transform from that unit to the
    /// corresponding SI unit (metre for `linear`, radian for `angular`, etc.).
    conv_factor: f64,
    /// PROJ short name, like `m`, `ft`, `us-ft`, etc... Might be NULL
    proj_short_name: String,
    /// Whether the object is deprecated
    deprecated: bool,
}
impl UnitInfo {
    pub fn new(
        auth_name: String,
        code: String,
        name: String,
        category: UnitCategory,
        conv_factor: f64,
        proj_short_name: String,
        deprecated: bool,
    ) -> Self {
        UnitInfo {
            auth_name,
            code,
            name,
            category,
            conv_factor,
            proj_short_name,
            deprecated,
        }
    }
}

/// Structure given description of a celestial body.
///
/// This structure may grow over time, and should not be directly allocated by
/// client code.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PROJ_CELESTIAL_BODY_INFO>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct CelestialBodyInfo {
    /// Authority name.
    auth_name: String,
    /// Object
    name: String,
}
impl CelestialBodyInfo {
    pub fn new(auth_name: String, name: String) -> Self { CelestialBodyInfo { auth_name, name } }
}
///Type of unit of measure.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UNIT_TYPE>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum UnitType {
    ///Angular unit of measure
    Angular = proj_sys::PJ_UNIT_TYPE_PJ_UT_ANGULAR,
    ///Linear unit of measure
    Linear = proj_sys::PJ_UNIT_TYPE_PJ_UT_LINEAR,
    ///Scale unit of measure
    Scale = proj_sys::PJ_UNIT_TYPE_PJ_UT_SCALE,
    ///Time unit of measure
    Time = proj_sys::PJ_UNIT_TYPE_PJ_UT_TIME,
    ///Parametric unit of measure
    Parametric = proj_sys::PJ_UNIT_TYPE_PJ_UT_PARAMETRIC,
}
///Type of Cartesian 2D coordinate system.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_CARTESIAN_CS_2D_TYPE>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum CartesianCs2dType {
    ///Easting-Norting
    EastingNorthing = proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_EASTING_NORTHING,
    ///Northing-Easting
    NorthingEasting = proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_NORTHING_EASTING,
    ///North Pole Easting/SOUTH-Norting/SOUTH
    NorthPoleEastingSouthNorthingSouth =
        proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_NORTH_POLE_EASTING_SOUTH_NORTHING_SOUTH,
    ///South Pole Easting/NORTH-Norting/NORTH
    SouthPoleEastingNorthNorthingNorth =
        proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_SOUTH_POLE_EASTING_NORTH_NORTHING_NORTH,
    ///Westing-southing
    WestingSouthing = proj_sys::PJ_CARTESIAN_CS_2D_TYPE_PJ_CART2D_WESTING_SOUTHING,
}
///Type of Ellipsoidal 2D coordinate system.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_ELLIPSOIDAL_CS_2D_TYPE>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum EllipsoidalCs2dType {
    ///Longitude-Latitude
    LongitudeLatitude = proj_sys::PJ_ELLIPSOIDAL_CS_2D_TYPE_PJ_ELLPS2D_LONGITUDE_LATITUDE,
    ///Latitude-Longitude
    LatitudeLongitude = proj_sys::PJ_ELLIPSOIDAL_CS_2D_TYPE_PJ_ELLPS2D_LATITUDE_LONGITUDE,
}
///Type of Ellipsoidal 3D coordinate system.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_ELLIPSOIDAL_CS_3D_TYPE>
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, TryFromPrimitive)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(u32)]
pub enum EllipsoidalCs3dType {
    ///Longitude-Latitude-Height(up)
    LongitudeLatitudeHeight,
    ///Latitude-Longitude-Height(up)
    LatitudeLongitudeHeight,
}
///Axis description.
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_AXIS_DESCRIPTION>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct AxisDescription {
    pub(crate) name: Option<String>,
    pub(crate) abbreviation: Option<String>,
    pub(crate) direction: AxisDirection,
    pub(crate) unit_name: Option<String>,
    pub(crate) unit_conv_factor: f64,
    pub(crate) unit_type: UnitType,
}
impl AxisDescription {
    pub fn new(
        name: Option<String>,
        abbreviation: Option<String>,
        direction: AxisDirection,
        unit_name: Option<String>,
        unit_conv_factor: f64,
        unit_type: UnitType,
    ) -> Result<Self, ProjError> {
        Ok(Self {
            name,
            abbreviation,
            direction: direction,
            unit_name: unit_name,
            unit_conv_factor,
            unit_type,
        })
    }
}
/// Description of a parameter value for a Conversion.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PARAM_DESCRIPTION>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct ParamDescription {
    name: Option<String>,
    auth_name: Option<String>,
    code: Option<String>,
    value: f64,
    unit_name: Option<String>,
    unit_conv_factor: f64,
    unit_type: UnitType,
}
impl ParamDescription {
    pub fn new(
        name: Option<String>,
        auth_name: Option<String>,
        code: Option<String>,
        value: f64,
        unit_name: Option<String>,
        unit_conv_factor: f64,
        unit_type: UnitType,
    ) -> Self {
        ParamDescription {
            name,
            auth_name,
            code,
            value,
            unit_name,
            unit_conv_factor,
            unit_type,
        }
    }
}

// region:internal
/// # References
///
/// * <https://github.com/OSGeo/PROJ/blob/master/src/iso19111/static.cpp>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(test, derive(strum::EnumIter))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, EnumString, strum::AsRefStr)]
pub enum AxisDirection {
    #[strum(serialize = "north")]
    North,
    #[strum(serialize = "northNorthEast")]
    NorthNorthEast,
    #[strum(serialize = "northEast")]
    NorthEast,
    #[strum(serialize = "eastNorthEast")]
    EastNorthEast,
    #[strum(serialize = "east")]
    East,
    #[strum(serialize = "eastSouthEast")]
    EastSouthEast,
    #[strum(serialize = "southEast")]
    SouthEast,
    #[strum(serialize = "southSouthEast")]
    SouthSouthEast,
    #[strum(serialize = "south")]
    South,
    #[strum(serialize = "southSouthWest")]
    SouthSouthWest,
    #[strum(serialize = "southWest")]
    SouthWest,
    #[strum(serialize = "westSouthWest")]
    WestSouthWest,
    #[strum(serialize = "west")]
    West,
    #[strum(serialize = "westNorthWest")]
    WestNorthWest,
    #[strum(serialize = "northWest")]
    NorthWest,
    #[strum(serialize = "northNorthWest")]
    NorthNorthWest,
    #[strum(serialize = "up")]
    Up,
    #[strum(serialize = "down")]
    Down,
    #[strum(serialize = "geocentricX")]
    GeocentricX,
    #[strum(serialize = "geocentricY")]
    GeocentricY,
    #[strum(serialize = "geocentricZ")]
    GeocentricZ,
    #[strum(serialize = "columnPositive")]
    ColumnPositive,
    #[strum(serialize = "columnNegative")]
    ColumnNegative,
    #[strum(serialize = "rowPositive")]
    RowPositive,
    #[strum(serialize = "rowNegative")]
    RowNegative,
    #[strum(serialize = "displayRight")]
    DisplayRight,
    #[strum(serialize = "displayLeft")]
    DisplayLeft,
    #[strum(serialize = "displayUp")]
    DisplayUp,
    #[strum(serialize = "displayDown")]
    DisplayDown,
    #[strum(serialize = "forward")]
    Forward,
    #[strum(serialize = "aft")]
    Aft,
    #[strum(serialize = "port")]
    Port,
    #[strum(serialize = "starboard")]
    Starboard,
    #[strum(serialize = "clockwise")]
    Clockwise,
    #[strum(serialize = "counterClockwise")]
    CounterClockwise,
    #[strum(serialize = "towards")]
    Towards,
    #[strum(serialize = "awayFrom")]
    AwayFrom,
    #[strum(serialize = "future")]
    Future,
    #[strum(serialize = "past")]
    Past,
    #[strum(serialize = "unspecified")]
    Unspecified,
}
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_cs_get_axis_info>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct AxisInfo {
    name: String,
    abbrev: String,
    direction: AxisDirection,
    unit_conv_factor: f64,
    unit_name: String,
    unit_auth_name: String,
    unit_code: String,
}
impl AxisInfo {
    pub fn new(
        name: String,
        abbrev: String,
        direction: AxisDirection,
        unit_conv_factor: f64,
        unit_name: String,
        unit_auth_name: String,
        unit_code: String,
    ) -> Self {
        AxisInfo {
            name,
            abbrev,
            direction,
            unit_conv_factor,
            unit_name,
            unit_auth_name,
            unit_code,
        }
    }
}
/// # References
///
/// * <https://github.com/OSGeo/PROJ/blob/master/src/proj.h>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct EllipsoidParameters {
    semi_major_metre: f64,
    semi_minor_metre: f64,
    is_semi_minor_computed: bool,
    inv_flattening: f64,
}
impl EllipsoidParameters {
    pub fn new(
        semi_major_metre: f64,
        semi_minor_metre: f64,
        is_semi_minor_computed: bool,
        inv_flattening: f64,
    ) -> Self {
        EllipsoidParameters {
            semi_major_metre,
            semi_minor_metre,
            is_semi_minor_computed,
            inv_flattening,
        }
    }
}
/// # References
///
/// * <https://github.com/OSGeo/PROJ/blob/master/src/proj.h>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct PrimeMeridianParameters {
    longitude: f64,
    unit_conv_factor: f64,
    unit_name: String,
}
impl PrimeMeridianParameters {
    pub fn new(longitude: f64, unit_conv_factor: f64, unit_name: String) -> Self {
        PrimeMeridianParameters {
            longitude,
            unit_conv_factor,
            unit_name,
        }
    }
}
#[doc = "# References"]
#[doc = "* <https://github.com/OSGeo/PROJ/blob/master/src/proj.h>"]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct CoordOperationMethodInfo {
    method_name: String,
    method_auth_name: String,
    method_code: String,
}
impl CoordOperationMethodInfo {
    pub fn new(method_name: String, method_auth_name: String, method_code: String) -> Self {
        CoordOperationMethodInfo {
            method_name,
            method_auth_name,
            method_code,
        }
    }
}
///# References
///
/// * <https://github.com/OSGeo/PROJ/blob/master/src/proj.h>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct CoordOperationParam {
    name: String,
    auth_name: String,
    code: String,
    value: f64,
    value_string: String,
    unit_conv_factor: f64,
    unit_name: String,
    unit_auth_name: String,
    unit_code: String,
    unit_category: UnitCategory,
}
impl CoordOperationParam {
    pub fn new(
        name: String,
        auth_name: String,
        code: String,
        value: f64,
        value_string: String,
        unit_conv_factor: f64,
        unit_name: String,
        unit_auth_name: String,
        unit_code: String,
        unit_category: UnitCategory,
    ) -> Self {
        CoordOperationParam {
            name,
            auth_name,
            code,
            value,
            value_string,
            unit_conv_factor,
            unit_name,
            unit_auth_name,
            unit_code,
            unit_category,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, EnumString, AsRefStr)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UnitCategory {
    #[strum(serialize = "unknown")]
    Unknown,
    #[strum(serialize = "none")]
    None,
    #[strum(serialize = "linear")]
    Linear,
    #[strum(serialize = "linear_per_time")]
    LinearPerTime,
    #[strum(serialize = "angular")]
    Angular,
    #[strum(serialize = "angular_per_time")]
    AngularPerTime,
    #[strum(serialize = "scale")]
    Scale,
    #[strum(serialize = "scale_per_time")]
    ScalePerTime,
    #[strum(serialize = "time")]
    Time,
    #[strum(serialize = "parametric")]
    Parametric,
    #[strum(serialize = "parametric_per_time")]
    ParametricPerTime,
}
/// # References
///
/// * <https://github.com/OSGeo/PROJ/blob/master/src/proj.h>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct CoordOperationGridUsed {
    short_name: String,
    full_name: String,
    package_name: String,
    url: String,
    direct_download: bool,
    open_license: bool,
    available: bool,
}
impl CoordOperationGridUsed {
    pub fn new(
        short_name: String,
        full_name: String,
        package_name: String,
        url: String,
        direct_download: bool,
        open_license: bool,
        available: bool,
    ) -> Self {
        CoordOperationGridUsed {
            short_name,
            full_name,
            package_name,
            url,
            direct_download,
            open_license,
            available,
        }
    }
}
/// # See Also
///
/// *[`crate::Context::get_database_metadata`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_database_metadata>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, AsRefStr)]
pub enum DatabaseMetadataKey {
    #[strum(serialize = "DATABASE.LAYOUT.VERSION.MAJOR")]
    DatabaseLayoutVersionMajor,
    #[strum(serialize = "DATABASE.LAYOUT.VERSION.MINOR")]
    DatabaseLayoutVersionMinor,
    #[strum(serialize = "EPSG.VERSION")]
    EpsgVersion,
    #[strum(serialize = "EPSG.DATE")]
    EpsgDate,
    #[strum(serialize = "ESRI.VERSION")]
    EsriVersion,
    #[strum(serialize = "ESRI.DATE")]
    EsriDate,
    #[strum(serialize = "IGNF.SOURCE")]
    IgnfSource,
    #[strum(serialize = "IGNF.VERSION")]
    IgnfVersion,
    #[strum(serialize = "IGNF.DATE")]
    IgnfDate,
    #[strum(serialize = "NKG.SOURCE")]
    NkgSource,
    #[strum(serialize = "NKG.VERSION")]
    NkgVersion,
    #[strum(serialize = "NKG.DATE")]
    NkgDate,
    #[strum(serialize = "PROJ.VERSION")]
    ProjVersion,
    #[strum(serialize = "PROJ_DATA.VERSION")]
    ProjDataVersion,
}
/// # See Also
///
/// *[`crate::Proj::crs_create_bound_crs_to_wgs84`]
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_bound_crs_to_WGS84>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(test, derive(strum::EnumIter))]
#[derive(Debug)]
pub enum AllowIntermediateCrs {
    Always,
    IfNoDirectTransformation,
    Never,
}
impl Display for AllowIntermediateCrs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            AllowIntermediateCrs::Always => "ALWAYS",
            AllowIntermediateCrs::IfNoDirectTransformation => "IF_NO_DIRECT_TRANSFORMATION",
            AllowIntermediateCrs::Never => "NEVER",
        };
        write!(f, "{text}")
    }
}
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_area_of_use>
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_area_of_ex>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct AreaOfUse {
    /// a string to receive the name of the area of use.
    area_name: String,
    /// a double to receive the west latitude (in degrees).
    west_lon_degree: f64,
    /// a double to receive the south latitude (in degrees).
    south_lat_degree: f64,
    /// a double to receive the east latitude (in degrees).
    east_lon_degree: f64,
    /// a double to receive the north latitude (in degrees).
    north_lat_degree: f64,
}
impl AreaOfUse {
    pub fn new(
        area_name: String,
        west_lon_degree: f64,
        south_lat_degree: f64,
        east_lon_degree: f64,
        north_lat_degree: f64,
    ) -> Self {
        AreaOfUse {
            area_name,
            west_lon_degree,
            south_lat_degree,
            east_lon_degree,
            north_lat_degree,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, EnumString)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum UomCategory {
    #[strum(serialize = "unknown")]
    Unknown,
    #[strum(serialize = "none")]
    None,
    #[strum(serialize = "linear")]
    Linear,
    #[strum(serialize = "linear_per_time")]
    LinearPerTime,
    #[strum(serialize = "angular")]
    Angular,
    #[strum(serialize = "angular_per_time")]
    AngularPerTime,
    #[strum(serialize = "scale")]
    Scale,
    #[strum(serialize = "scale_per_time")]
    ScalePerTime,
    #[strum(serialize = "time")]
    Time,
    #[strum(serialize = "parametric")]
    Parametric,
    #[strum(serialize = "parametric_per_time")]
    ParametricPerTime,
}
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_uom_get_info_from_database>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct UomInfo {
    name: String,
    /// a double to receive the conversion factor of the unit to the
    /// corresponding SI unit (metre for `linear`, radian for `angular`, etc.).
    conv_factor: f64,
    category: UomCategory,
}
impl UomInfo {
    pub fn new(name: String, conv_factor: f64, category: UomCategory) -> Self {
        UomInfo {
            name,
            conv_factor,
            category,
        }
    }
}
///# References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_uom_get_info_from_database>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct GridInfoDB {
    /// a string value to store the grid full filename.
    full_name: String,
    /// a string value to store the package name where the grid might be found.
    package_name: String,
    /// a string value to store the grid URL or the package URL where the grid
    /// might be found.
    url: String,
    /// a boolean value to store whether *out_url can be downloaded directly.
    direct_download: bool,
    /// a boolean value to store whether the grid is released with an open
    /// license.
    open_license: bool,
    /// a boolean value to store whether the grid is available at runtime.
    available: bool,
}
impl GridInfoDB {
    pub fn new(
        full_name: String,
        package_name: String,
        url: String,
        direct_download: bool,
        open_license: bool,
        available: bool,
    ) -> Self {
        GridInfoDB {
            full_name,
            package_name,
            url,
            direct_download,
            open_license,
            available,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InsertObjectSession {
    pub(crate) arc_ctx_ptr: Arc<ContextPtr>,
    pub(crate) ptr: *mut proj_sys::PJ_INSERT_SESSION,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OperationFactoryContext {
    pub(crate) arc_ctx_ptr: Arc<ContextPtr>,
    pub(crate) ptr: *mut proj_sys::PJ_OPERATION_FACTORY_CONTEXT,
}
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProjObjList {
    pub(crate) arc_ctx_ptr: Arc<ContextPtr>,
    ptr: *mut proj_sys::PJ_OBJ_LIST,
    count: usize,
    pub(crate) _owned_cstrings: OwnedCStrings,
}
impl ProjObjList {
    pub(crate) fn new(
        ctx_ptr: Arc<ContextPtr>,
        ptr: *mut proj_sys::PJ_OBJ_LIST,
    ) -> Result<ProjObjList, ProjError> {
        check_result!(ptr.is_null(), "PJ_OBJ_LIST pointer is null.");
        let count = unsafe { proj_sys::proj_list_get_count(ptr) };
        check_result!(count < 1, "PJ_OBJ_LIST count 0.");
        clerk::debug!("pj_obj_list count: {count}");
        Ok(ProjObjList {
            arc_ctx_ptr: ctx_ptr,
            ptr,
            count: count as usize,
            _owned_cstrings: OwnedCStrings::new(),
        })
    }

    /// Create a `ProjObjList` object from pointer, panic if pointer is null.
    pub(crate) fn new_with_owned_cstrings(
        ctx_ptr: Arc<ContextPtr>,
        ptr: *mut proj_sys::PJ_OBJ_LIST,
        owned_cstrings: OwnedCStrings,
    ) -> Result<ProjObjList, ProjError> {
        check_result!(ptr.is_null(), "PJ_OBJ_LIST pointer is null.");
        let count = unsafe { proj_sys::proj_list_get_count(ptr) };
        check_result!(count < 1, "PJ_OBJ_LIST count 0.");
        clerk::debug!("pj_obj_list count: {count}");
        Ok(ProjObjList {
            arc_ctx_ptr: ctx_ptr,
            ptr,
            count: count as usize,
            _owned_cstrings: owned_cstrings,
        })
    }

    pub(crate) fn ptr(&self) -> *mut proj_sys::PJ_OBJ_LIST { self.ptr }
    ///Return an object from the result set.
    ///
    /// # References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_get>
    pub fn get(&self, index: usize) -> Result<Proj, ProjError> {
        check_result!(index > self.count, "Error");
        let ptr =
            unsafe { proj_sys::proj_list_get(self.arc_ctx_ptr.ptr(), self.ptr, index as i32) };

        Proj::new_with_owned_cstrings(self.arc_ctx_ptr.clone(), ptr, self._owned_cstrings.clone())
    }
    ///Return the number of objects in the result set.
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_get_count>
    pub fn get_count(&self) -> usize { self.count }
}
