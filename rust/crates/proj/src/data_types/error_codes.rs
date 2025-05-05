pub(crate) enum PjErrorCode {
    ProjSuccess,
    //Errors in class PROJ_ERR_INVALID_OP
    InvalidOp,
    InvalidOpWrongSyntax,
    InvalidOpMissingArg,
    InvalidOpIllegalArgValue,
    InvalidOpMutuallyExclusiveArgs,
    InvalidOpFileNotFoundOrInvalid,
    //Errors in class PROJ_ERR_COORD_TRANSFM
    CoordTransfm,
    CoordTransfmInvalidCoord,
    CoordTransfmOutsideProjectionDomain,
    CoordTransfmNoOperation,
    CoordTransfmOutsideGrid,
    CoordTransfmGridAtNodata,
    CoordTransfmNoConvergence,
    CoordTransfmMissingTime,
    //Errors in class PROJ_ERR_OTHER
    Other,
    OtherApiMisuse,
    OtherNoInverseOp,
    OtherNetworkError,
}

impl From<u32> for PjErrorCode {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::ProjSuccess,
            //Errors in class PROJ_ERR_INVALID_OP
            proj_sys::PROJ_ERR_INVALID_OP => Self::InvalidOp,
            proj_sys::PROJ_ERR_INVALID_OP_WRONG_SYNTAX => Self::InvalidOpWrongSyntax,
            proj_sys::PROJ_ERR_INVALID_OP_MISSING_ARG => Self::InvalidOpMissingArg,
            proj_sys::PROJ_ERR_INVALID_OP_ILLEGAL_ARG_VALUE => Self::InvalidOpIllegalArgValue,
            proj_sys::PROJ_ERR_INVALID_OP_MUTUALLY_EXCLUSIVE_ARGS => {
                Self::InvalidOpMutuallyExclusiveArgs
            }
            proj_sys::PROJ_ERR_INVALID_OP_FILE_NOT_FOUND_OR_INVALID => {
                Self::InvalidOpFileNotFoundOrInvalid
            }
            //Errors in class PROJ_ERR_COORD_TRANSFM
            proj_sys::PROJ_ERR_COORD_TRANSFM => Self::CoordTransfm,
            proj_sys::PROJ_ERR_COORD_TRANSFM_INVALID_COORD => Self::CoordTransfmInvalidCoord,
            proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_PROJECTION_DOMAIN => {
                Self::CoordTransfmOutsideProjectionDomain
            }
            proj_sys::PROJ_ERR_COORD_TRANSFM_NO_OPERATION => Self::CoordTransfmNoOperation,
            proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_GRID => Self::CoordTransfmOutsideGrid,
            proj_sys::PROJ_ERR_COORD_TRANSFM_GRID_AT_NODATA => Self::CoordTransfmGridAtNodata,
            proj_sys::PROJ_ERR_COORD_TRANSFM_NO_CONVERGENCE => Self::CoordTransfmNoConvergence,
            proj_sys::PROJ_ERR_COORD_TRANSFM_MISSING_TIME => Self::CoordTransfmMissingTime,
            //Errors in class PROJ_ERR_OTHER
            proj_sys::PROJ_ERR_OTHER => Self::Other,
            proj_sys::PROJ_ERR_OTHER_API_MISUSE => Self::OtherApiMisuse,
            proj_sys::PROJ_ERR_OTHER_NO_INVERSE_OP => Self::OtherNoInverseOp,
            proj_sys::PROJ_ERR_OTHER_NETWORK_ERROR => Self::OtherNetworkError,

            code => panic!("Unknown error: {code}"),
        }
    }
}

impl From<&PjErrorCode> for i32 {
    fn from(value: &PjErrorCode) -> Self {
        match value {
            PjErrorCode::ProjSuccess => 0,
            //Errors in class PROJ_ERR_INVALID_OP
            PjErrorCode::InvalidOp => proj_sys::PROJ_ERR_INVALID_OP as i32,
            PjErrorCode::InvalidOpWrongSyntax => proj_sys::PROJ_ERR_INVALID_OP_WRONG_SYNTAX as i32,
            PjErrorCode::InvalidOpMissingArg => proj_sys::PROJ_ERR_INVALID_OP_MISSING_ARG as i32,
            PjErrorCode::InvalidOpIllegalArgValue => {
                proj_sys::PROJ_ERR_INVALID_OP_ILLEGAL_ARG_VALUE as i32
            }
            PjErrorCode::InvalidOpMutuallyExclusiveArgs => {
                proj_sys::PROJ_ERR_INVALID_OP_MUTUALLY_EXCLUSIVE_ARGS as i32
            }
            PjErrorCode::InvalidOpFileNotFoundOrInvalid => {
                proj_sys::PROJ_ERR_INVALID_OP_FILE_NOT_FOUND_OR_INVALID as i32
            }
            //Errors in class PROJ_ERR_COORD_TRANSFM
            PjErrorCode::CoordTransfm => proj_sys::PROJ_ERR_COORD_TRANSFM as i32,
            PjErrorCode::CoordTransfmInvalidCoord => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_INVALID_COORD as i32
            }
            PjErrorCode::CoordTransfmOutsideProjectionDomain => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_PROJECTION_DOMAIN as i32
            }
            PjErrorCode::CoordTransfmNoOperation => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_NO_OPERATION as i32
            }
            PjErrorCode::CoordTransfmOutsideGrid => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_GRID as i32
            }
            PjErrorCode::CoordTransfmGridAtNodata => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_GRID_AT_NODATA as i32
            }
            PjErrorCode::CoordTransfmNoConvergence => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_NO_CONVERGENCE as i32
            }
            PjErrorCode::CoordTransfmMissingTime => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_MISSING_TIME as i32
            }
            //Errors in class PROJ_ERR_OTHER
            PjErrorCode::Other => proj_sys::PROJ_ERR_OTHER as i32,
            PjErrorCode::OtherApiMisuse => proj_sys::PROJ_ERR_OTHER_API_MISUSE as i32,
            PjErrorCode::OtherNoInverseOp => proj_sys::PROJ_ERR_OTHER_NO_INVERSE_OP as i32,
            PjErrorCode::OtherNetworkError => proj_sys::PROJ_ERR_OTHER_NETWORK_ERROR as i32,
        }
    }
}
