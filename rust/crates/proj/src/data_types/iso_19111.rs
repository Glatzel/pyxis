enum _PjGuessedWktDialect {
    Wkt2_2019,
    Wkt2_2018,
    Wkt2_2015,
    Wkt1Gdal,
    Wkt1Esri,
    NotWkt,
}
enum _PjCategory {
    Ellipsoid,
    PrimeMeridian,
    Datum,
    Crs,
    CoordinateOperation,
    DatumEnsemble,
}
enum _PjType {
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
enum _PjComparisonCriterion {
    Strict,
    Equivalent,
    EquivalentExceptAxisOrderGeogcrs,
}
enum _PjWktType {
    Wkt2_2015,
    Wkt2_2015Simplified,
    Wkt2_2019,
    Wkt2_2018,
    Wkt2_2019Simplified,
    Wkt2_2018Simplified,
    Wkt1Gdal,
    Wkt1Esri,
}
enum _ProjCrsExtentUse {
    None,
    Both,
    Intersection,
    Smallest,
}
enum _ProjGridAvailabilityUse {
    UsedForSorting,
    DiscardOperationIfMissingGrid,
    Ignored,
    KnownAvailable,
}

enum _PjProjStringType {
    Proj5,
    Proj4,
}

enum _ProjSpatialCriterion {
    StrictContainment,
    PartialIntersection,
}
enum _ProjIntermediateCrsUse {
    Always,
    IfNoDirectTransformation,
    Never,
}

enum _PjCoordinateSystemType {
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

enum _PjUnitType {
    Angular,
    Linear,
    Scale,
    Time,
    Parametric,
}

enum _PjCartesianCs2dType {
    EastingNorthing,
    NorthingEasting,
    NorthPoleEastingSouthNorthingSouth,
    SouthPoleEastingNorthNorthingNorth,
    WestingSouthing,
}

enum _PjEllipsoidalCs2dType {
    LongitudeLatitude,
    LatitudeLongitude,
}

enum _PjEllipsoidalCs3dType {
    LongitudeLatitudeHeight,
    LatitudeLongitudeHeight,
}

// PJ_AXIS_DESCRIPTION

// PJ_PARAM_DESCRIPTION
