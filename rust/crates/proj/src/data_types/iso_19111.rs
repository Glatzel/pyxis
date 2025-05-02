pub enum PjGuessedWktDialect {
    PjGuessedWkt2_2019,
    PjGuessedWkt2_2018,
    PjGuessedWkt2_2015,
    PjGuessedWkt1Gdal,
    PjGuessedWkt1Esri,
    PjGuessedNotWkt,
}
pub enum PjCategory {
    PjCategoryEllipsoid,
    PjCategoryPrimeMeridian,
    PjCategoryDatum,
    PjCategoryCrs,
    PjCategoryCoordinateOperation,
    PjCategoryDatumEnsemble,
}
pub enum PjType {
    PjTypeUnknown,
    PjTypeEllipsoid,
    PjTypePrimeMeridian,
    PjTypeGeodeticReferenceFrame,
    PjTypeDynamicGeodeticReferenceFrame,
    PjTypeVerticalReferenceFrame,
    PjTypeDynamicVerticalReferenceFrame,
    PjTypeDatumEnsemble,
    PjTypeCrs,
    PjTypeGeodeticCrs,
    PjTypeGeocentricCrs,
    PjTypeGeographicCr,
    PjTypeGeographic2dCrs,
    PjTypeGeographic3dCrs,
    PjTypeVerticalCrs,
    PjTypeProjectedCrs,
    PjTypeCompoundCrs,
    PjTypeTemporalCrs,
    PjTypeEngineeringCrs,
    PjTypeBoundCrs,
    PjTypeOtherCrs,
    PjTypeConversion,
    PjTypeTransformation,
    PjTypeConcatenatedOperation,
    PjTypeOtherCoordinateOperation,
    PjTypeTemporalDatum,
    PjTypeEngineeringDatum,
    PjTypeParametricDatum,
    PjTypeDerivedProjectedCrs,
    PjTypeCoordinateMetadata,
}
pub enum PjComparisonCriterion {
    PjCompStrict,
    PjCompEquivalent,
    PjCompEquivalentExceptAxisOrderGeogcrs,
}
pub enum PjWktType {
    PjWkt2_2015,
    PjWkt2_2015Simplified,
    PjWkt2_2019,
    PjWkt2_2018,
    PjWkt2_2019Simplified,
    PjWkt2_2018Simplified,
    PjWkt1Gdal,
    PjWkt1Esri,
}
pub enum ProjCrsExtentUse {
    PjCrsExtentNone,
    PjCrsExtentBoth,
    PjCrsExtentIntersection,
    PjCrsExtentSmallest,
}
pub enum ProjGridAvailabilityUse {
    ProjGridAvailabilityUsedForSorting,
    ProjGridAvailabilityDiscardOperationIfMissingGrid,
    ProjGridAvailabilityIgnored,
    ProjGridAvailabilityKnownAvailable,
}

pub enum PjProjStringType {
    PjProj5,
    PjProj4,
}

pub enum ProjSpatialCriterion {
    ProjSpatialCriterionStrictContainment,
    ProjSpatialCriterionPartialIntersection,
}
pub enum ProjIntermediateCrsUse {
    ProjIntermediateCrsUseAlways,
    ProjIntermediateCrsUseIfNoDirectTransformation,
    ProjIntermediateCrsUseNever,
}

pub enum PjCoordinateSystemType {
    PjCsTypeUnknown,
    PjCsTypeCartesian,
    PjCsTypeEllipsoidal,
    PjCsTypeVertical,
    PjCsTypeSpherical,
    PjCsTypeOrdinal,
    PjCsTypeParametric,
    PjCsTypeDatetimetemporal,
    PjCsTypeTemporalcount,
    PjCsTypeTemporalmeasure,
}

// pub struct PROJ_CRS_INFO{}

// pub struct PROJ_CRS_LIST_PARAMETERS{}

// pub struct PROJ_UNIT_INFO

// pub struct PROJ_CELESTIAL_BODY_INFO

pub enum PjUnitType {
    PjUtAngular,
    PjUtLinear,
    PjUtScale,
    PjUtTime,
    PjUtParametric,
}

pub enum PjCartesianCs2dType {
    PjCart2dEastingNorthing,
    PjCart2dNorthingEasting,
    PjCart2dNorthPoleEastingSouthNorthingSouth,
    PjCart2dSouthPoleEastingNorthNorthingNorth,
    PjCart2dWestingSouthing,
}

pub enum PjEllipsoidalCs2dType {
    PjEllps2dLongitudeLatitude,
    PjEllps2dLatitudeLongitude,
}

pub enum PjEllipsoidalCs3dType {
    PjEllps3dLongitudeLatitudeHeight,
    PjEllps3dLatitudeLongitudeHeight,
}

// PJ_AXIS_DESCRIPTION

// PJ_PARAM_DESCRIPTION
