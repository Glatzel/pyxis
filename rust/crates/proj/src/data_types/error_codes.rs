pub(crate) enum PjErrorCode {
    //Errors in class PROJ_ERR_INVALID_OP
    ProjErrInvalidOp,
    ProjErrInvalidOpWrongSyntax,
    ProjErrInvalidOpMissingArg,
    ProjErrInvalidOpIllegalArgValue,
    ProjErrInvalidOpMutuallyExclusiveArgs,
    ProjErrInvalidOpFileNotFoundOrInvalid,
    //Errors in class PROJ_ERR_COORD_TRANSFM
    ProjErrCoordTransfm,
    ProjErrCoordTransfmInvalidCoord,
    ProjErrCoordTransfmOutsideProjectionDomain,
    ProjErrCoordTransfmNoOperation,
    ProjErrCoordTransfmOutsideGrid,
    ProjErrCoordTransfmGridAtNodata,
    ProjErrCoordTransfmNoConvergence,
    ProjErrCoordTransfmMissingTime,
    //Errors in class PROJ_ERR_OTHER
    ProjErrOther,
    ProjErrOtherApiMisuse,
    ProjErrOtherNoInverseOp,
    ProjErrOtherNetworkError,
}
impl From<u32> for PjErrorCode {
    fn from(value: u32) -> Self {
        match value {
            //Errors in class PROJ_ERR_INVALID_OP
            proj_sys::PROJ_ERR_INVALID_OP => Self::ProjErrInvalidOp,
            proj_sys::PROJ_ERR_INVALID_OP_WRONG_SYNTAX => Self::ProjErrInvalidOpWrongSyntax,
            proj_sys::PROJ_ERR_INVALID_OP_MISSING_ARG => Self::ProjErrInvalidOpMissingArg,
            proj_sys::PROJ_ERR_INVALID_OP_ILLEGAL_ARG_VALUE => {
                Self::ProjErrInvalidOpIllegalArgValue
            }
            proj_sys::PROJ_ERR_INVALID_OP_MUTUALLY_EXCLUSIVE_ARGS => {
                Self::ProjErrInvalidOpMutuallyExclusiveArgs
            }
            proj_sys::PROJ_ERR_INVALID_OP_FILE_NOT_FOUND_OR_INVALID => {
                Self::ProjErrInvalidOpFileNotFoundOrInvalid
            }
            //Errors in class PROJ_ERR_COORD_TRANSFM
            proj_sys::PROJ_ERR_COORD_TRANSFM => Self::ProjErrCoordTransfm,
            proj_sys::PROJ_ERR_COORD_TRANSFM_INVALID_COORD => Self::ProjErrCoordTransfmInvalidCoord,
            proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_PROJECTION_DOMAIN => {
                Self::ProjErrCoordTransfmOutsideProjectionDomain
            }
            proj_sys::PROJ_ERR_COORD_TRANSFM_NO_OPERATION => Self::ProjErrCoordTransfmNoOperation,
            proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_GRID => Self::ProjErrCoordTransfmOutsideGrid,
            proj_sys::PROJ_ERR_COORD_TRANSFM_GRID_AT_NODATA => {
                Self::ProjErrCoordTransfmGridAtNodata
            }
            proj_sys::PROJ_ERR_COORD_TRANSFM_NO_CONVERGENCE => {
                Self::ProjErrCoordTransfmNoConvergence
            }
            proj_sys::PROJ_ERR_COORD_TRANSFM_MISSING_TIME => Self::ProjErrCoordTransfmMissingTime,
            //Errors in class PROJ_ERR_OTHER
            proj_sys::PROJ_ERR_OTHER => Self::ProjErrOther,
            proj_sys::PROJ_ERR_OTHER_API_MISUSE => Self::ProjErrOtherApiMisuse,
            proj_sys::PROJ_ERR_OTHER_NO_INVERSE_OP => Self::ProjErrOtherNoInverseOp,
            proj_sys::PROJ_ERR_OTHER_NETWORK_ERROR => Self::ProjErrOtherNetworkError,

            code => panic!("Unknown error: {code}"),
        }
    }
}
impl From<PjErrorCode> for i32 {
    fn from(value: PjErrorCode) -> Self {
        match value {
            //Errors in class PROJ_ERR_INVALID_OP
            PjErrorCode::ProjErrInvalidOp => proj_sys::PROJ_ERR_INVALID_OP as i32,
            PjErrorCode::ProjErrInvalidOpWrongSyntax => {
                proj_sys::PROJ_ERR_INVALID_OP_WRONG_SYNTAX as i32
            }
            PjErrorCode::ProjErrInvalidOpMissingArg => {
                proj_sys::PROJ_ERR_INVALID_OP_MISSING_ARG as i32
            }
            PjErrorCode::ProjErrInvalidOpIllegalArgValue => {
                proj_sys::PROJ_ERR_INVALID_OP_ILLEGAL_ARG_VALUE as i32
            }
            PjErrorCode::ProjErrInvalidOpMutuallyExclusiveArgs => {
                proj_sys::PROJ_ERR_INVALID_OP_MUTUALLY_EXCLUSIVE_ARGS as i32
            }
            PjErrorCode::ProjErrInvalidOpFileNotFoundOrInvalid => {
                proj_sys::PROJ_ERR_INVALID_OP_FILE_NOT_FOUND_OR_INVALID as i32
            }
            //Errors in class PROJ_ERR_COORD_TRANSFM
            PjErrorCode::ProjErrCoordTransfm => proj_sys::PROJ_ERR_COORD_TRANSFM as i32,
            PjErrorCode::ProjErrCoordTransfmInvalidCoord => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_INVALID_COORD as i32
            }
            PjErrorCode::ProjErrCoordTransfmOutsideProjectionDomain => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_PROJECTION_DOMAIN as i32
            }
            PjErrorCode::ProjErrCoordTransfmNoOperation => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_NO_OPERATION as i32
            }
            PjErrorCode::ProjErrCoordTransfmOutsideGrid => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_OUTSIDE_GRID as i32
            }
            PjErrorCode::ProjErrCoordTransfmGridAtNodata => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_GRID_AT_NODATA as i32
            }
            PjErrorCode::ProjErrCoordTransfmNoConvergence => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_NO_CONVERGENCE as i32
            }
            PjErrorCode::ProjErrCoordTransfmMissingTime => {
                proj_sys::PROJ_ERR_COORD_TRANSFM_MISSING_TIME as i32
            }
            //Errors in class PROJ_ERR_OTHER
            PjErrorCode::ProjErrOther => proj_sys::PROJ_ERR_OTHER as i32,
            PjErrorCode::ProjErrOtherApiMisuse => proj_sys::PROJ_ERR_OTHER_API_MISUSE as i32,
            PjErrorCode::ProjErrOtherNoInverseOp => proj_sys::PROJ_ERR_OTHER_NO_INVERSE_OP as i32,
            PjErrorCode::ProjErrOtherNetworkError => proj_sys::PROJ_ERR_OTHER_NETWORK_ERROR as i32,
        }
    }
}
