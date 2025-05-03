pub enum PjGuessedWktDialect {
    Wkt2_2019,
    Wkt2_2018,
    Wkt2_2015,
    Wkt1Gdal,
    Wkt1Esri,
    NotWkt,
}
pub enum PjCategory {
    Ellipsoid,
    PrimeMeridian,
    Datum,
    Crs,
    CoordinateOperation,
    DatumEnsemble,
}
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
pub enum PjComparisonCriterion {
    Strict,
    Equivalent,
    EquivalentExceptAxisOrderGeogcrs,
}
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
pub enum ProjCrsExtentUse {
    None,
    Both,
    Intersection,
    Smallest,
}
pub enum ProjGridAvailabilityUse {
    UsedForSorting,
    DiscardOperationIfMissingGrid,
    Ignored,
    KnownAvailable,
}

pub enum PjProjStringType {
    Proj5,
    Proj4,
}

pub enum ProjSpatialCriterion {
    StrictContainment,
    PartialIntersection,
}
pub enum ProjIntermediateCrsUse {
    Always,
    IfNoDirectTransformation,
    Never,
}

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

// pub struct PROJ_CRS_INFO{}

// pub struct PROJ_CRS_LIST_PARAMETERS{}

// pub struct PROJ_UNIT_INFO

// pub struct PROJ_CELESTIAL_BODY_INFO

pub enum PjUnitType {
    Angular,
    Linear,
    Scale,
    Time,
    Parametric,
}

pub enum PjCartesianCs2dType {
    EastingNorthing,
    NorthingEasting,
    NorthPoleEastingSouthNorthingSouth,
    SouthPoleEastingNorthNorthingNorth,
    WestingSouthing,
}

pub enum PjEllipsoidalCs2dType {
    LongitudeLatitude,
    LatitudeLongitude,
}

pub enum PjEllipsoidalCs3dType {
    LongitudeLatitudeHeight,
    LatitudeLongitudeHeight,
}

// PJ_AXIS_DESCRIPTION

// PJ_PARAM_DESCRIPTION
