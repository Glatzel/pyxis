use num_enum::{FromPrimitive, IntoPrimitive};

#[derive(Debug, Clone, FromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub(crate) enum PjError {
    Success = 0,
    //Errors in class PROJ_ERR_INVALID_OP
    InvalidOp = proj_sys::PROJ_ERR_INVALID_OP,
    InvalidOpWrongSyntax = proj_sys::PROJ_ERR_INVALID_OP_WRONG_SYNTAX,
    InvalidOpMissingArg = proj_sys::PROJ_ERR_INVALID_OP_MISSING_ARG,
    InvalidOpIllegalArgValue = proj_sys::PROJ_ERR_INVALID_OP_ILLEGAL_ARG_VALUE,
    InvalidOpMutuallyExclusiveArgs = proj_sys::PROJ_ERR_INVALID_OP_MUTUALLY_EXCLUSIVE_ARGS,
    InvalidOpFileNotFoundOrInvalid = proj_sys::PROJ_ERR_INVALID_OP_FILE_NOT_FOUND_OR_INVALID,
    //Errors in class PROJ_ERR_COORD_TRANSFM
    CoordTransfm = proj_sys::PROJ_ERR_COORD_TRANSFM,
    CoordTransfmInvalidCoord = proj_sys::PROJ_ERR_COORD_TRANSFM_INVALID_COORD,
    CoordTransfmOutsideProjectionDomain =
        proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_PROJECTION_DOMAIN,
    CoordTransfmNoOperation = proj_sys::PROJ_ERR_COORD_TRANSFM_NO_OPERATION,
    CoordTransfmOutsideGrid = proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_GRID,
    CoordTransfmGridAtNodata = proj_sys::PROJ_ERR_COORD_TRANSFM_GRID_AT_NODATA,
    CoordTransfmNoConvergence = proj_sys::PROJ_ERR_COORD_TRANSFM_NO_CONVERGENCE,
    CoordTransfmMissingTime = proj_sys::PROJ_ERR_COORD_TRANSFM_MISSING_TIME,
    //Errors in class PROJ_ERR_OTHER
    #[num_enum(default)]
    Other = proj_sys::PROJ_ERR_OTHER,
    OtherApiMisuse = proj_sys::PROJ_ERR_OTHER_API_MISUSE,
    OtherNoInverseOp = proj_sys::PROJ_ERR_OTHER_NO_INVERSE_OP,
    OtherNetworkError = proj_sys::PROJ_ERR_OTHER_NETWORK_ERROR,
}
impl From<PjError> for i32 {
    fn from(value: PjError) -> Self { u32::from(value) as i32 }
}

impl From<&PjError> for i32 {
    fn from(value: &PjError) -> Self { Into::<u32>::into(value.clone()) as i32 }
}
impl From<i32> for PjError {
    fn from(value: i32) -> PjError { PjError::from(value as u32) }
}
