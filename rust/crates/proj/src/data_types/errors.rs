use num_enum::{FromPrimitive, IntoPrimitive};

///Three classes of errors are defined below. The belonging of a given error
/// code to a class can bit tested with a binary and test. The error class
/// itself can be used as an error value in some rare cases where the error does
/// not fit into a more precise error value.
///
///Those error codes are still quite generic for a number of them. Details on
/// the actual errors will be typically logged with the PJ_LOG_ERROR level.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#error-codes>
#[derive(Debug, Clone, FromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub(crate) enum ProjError {
    Success = 0,

    //Errors in class PROJ_ERR_INVALID_OP
    /// Class of error codes typically related to coordinate operation
    /// initialization, typically when creating a PJ* object from a PROJ
    /// string.
    InvalidOp = proj_sys::PROJ_ERR_INVALID_OP,
    ///Invalid pipeline structure, missing +proj argument, etc.
    InvalidOpWrongSyntax = proj_sys::PROJ_ERR_INVALID_OP_WRONG_SYNTAX,
    ///Missing required operation parameter
    InvalidOpMissingArg = proj_sys::PROJ_ERR_INVALID_OP_MISSING_ARG,
    ///One of the operation parameter has an illegal value.
    InvalidOpIllegalArgValue = proj_sys::PROJ_ERR_INVALID_OP_ILLEGAL_ARG_VALUE,
    ///Mutually exclusive arguments
    InvalidOpMutuallyExclusiveArgs = proj_sys::PROJ_ERR_INVALID_OP_MUTUALLY_EXCLUSIVE_ARGS,
    ///File not found or with invalid content (particular case of
    /// PROJ_ERR_INVALID_OP_ILLEGAL_ARG_VALUE)
    InvalidOpFileNotFoundOrInvalid = proj_sys::PROJ_ERR_INVALID_OP_FILE_NOT_FOUND_OR_INVALID,

    //Errors in class PROJ_ERR_COORD_TRANSFM
    ///Class of error codes related to transformation on a specific coordinate.
    CoordTransfm = proj_sys::PROJ_ERR_COORD_TRANSFM,
    ///Invalid input coordinate. e.g. a latitude > 90°.
    CoordTransfmInvalidCoord = proj_sys::PROJ_ERR_COORD_TRANSFM_INVALID_COORD,
    ///Coordinate is outside of the projection domain. e.g. approximate
    /// mercator with |longitude - lon_0| > 90°, or iterative convergence method
    /// failed.
    CoordTransfmOutsideProjectionDomain =
        proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_PROJECTION_DOMAIN,
    ///No operation found, e.g. if no match the required accuracy, or if
    /// ballpark transformations were asked to not be used and they would be
    /// only such candidate.
    CoordTransfmNoOperation = proj_sys::PROJ_ERR_COORD_TRANSFM_NO_OPERATION,
    ///Point to transform falls outside grid/subgrid/TIN.
    CoordTransfmOutsideGrid = proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_GRID,
    ///Point to transform falls in a grid cell that evaluates to nodata.
    CoordTransfmGridAtNodata = proj_sys::PROJ_ERR_COORD_TRANSFM_GRID_AT_NODATA,
    ///Point to transform falls in a grid cell that evaluates to nodata.
    CoordTransfmNoConvergence = proj_sys::PROJ_ERR_COORD_TRANSFM_NO_CONVERGENCE,
    CoordTransfmMissingTime = proj_sys::PROJ_ERR_COORD_TRANSFM_MISSING_TIME,

    //Errors in class PROJ_ERR_OTHER
    ///Class of error codes that do not fit into one of the above class.
    #[num_enum(default)]
    Other = proj_sys::PROJ_ERR_OTHER,
    ///Error related to a misuse of PROJ API.
    OtherApiMisuse = proj_sys::PROJ_ERR_OTHER_API_MISUSE,
    ///No inverse method available
    OtherNoInverseOp = proj_sys::PROJ_ERR_OTHER_NO_INVERSE_OP,
    ///Failure when accessing a network resource.
    OtherNetworkError = proj_sys::PROJ_ERR_OTHER_NETWORK_ERROR,
}
impl From<ProjError> for i32 {
    fn from(value: ProjError) -> Self { u32::from(value) as i32 }
}
impl From<&ProjError> for i32 {
    fn from(value: &ProjError) -> Self { Into::<u32>::into(value.clone()) as i32 }
}
impl From<i32> for ProjError {
    fn from(value: i32) -> ProjError { ProjError::from(value as u32) }
}
