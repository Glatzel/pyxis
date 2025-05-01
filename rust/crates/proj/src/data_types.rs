// region:Transformation objects
pub struct Pj {
    pj: *mut proj_sys::PJ,
}
impl Drop for Pj {
    fn drop(&mut self) {
        unsafe { proj_sys::proj_destroy(self.pj) };
    }
}
// region:Coordinate transformation
///https://proj.org/en/stable/development/reference/functions.html#coordinate-transformation
impl Pj {
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans
    pub fn trans(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_get_last_used_operation
    pub fn trans_get_last_used_operation(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_generic
    pub fn proj_trans_generic(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_array
    pub fn proj_trans_array(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds
    pub fn proj_trans_bounds(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds_3D
    pub fn proj_trans_bounds_3d(&self) {
        unimplemented!()
    }
}
impl Pj {
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_errno
    fn _errno(&self) -> PjErrorCode {
        PjErrorCode::from(unsafe { proj_sys::proj_errno(self.pj) } as u32)
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_set
    fn _errno_set(&self, err: PjErrorCode) -> &Self {
        unsafe { proj_sys::proj_errno_set(self.pj, i32::from(err)) };
        self
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_reset
    fn _errno_reset(&self) -> PjErrorCode {
        PjErrorCode::from(unsafe { proj_sys::proj_errno_reset(self.pj) } as u32)
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_errno_restore
    fn _errno_restore(&self, err: PjErrorCode) -> &Self {
        unsafe { proj_sys::proj_errno_restore(self.pj, i32::from(err)) };
        self
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno_string
    fn _errno_string(&self, err: PjErrorCode) -> String {
        crate::c_char_to_string(unsafe { proj_sys::proj_errno_string(i32::from(err)) })
    }
}

struct _PjDirection {}

pub struct PjContext {
    ctx: *mut proj_sys::PJ_CONTEXT,
}
// region:Threading contexts
impl PjContext {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_create
    pub fn new() -> Self {
        Self {
            ctx: unsafe { proj_sys::proj_context_create() },
        }
    }
}
impl Clone for PjContext {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_clone
    fn clone(&self) -> Self {
        Self {
            ctx: unsafe { proj_sys::proj_context_clone(self.ctx) },
        }
    }
}
impl Drop for PjContext {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_destroy
    fn drop(&mut self) {
        unsafe { proj_sys::proj_context_destroy(self.ctx) };
    }
}
// region:Transformation setup
///https://proj.org/en/stable/development/reference/functions.html#transformation-setup
impl PjContext {
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_create
    pub fn proj_create(&self, definition: &str) -> Pj {
        Pj {
            pj: unsafe { proj_sys::proj_create(self.ctx, definition.as_ptr() as *const i8) },
        }
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv
    pub fn proj_create_argv(&self, definition: &[&str]) -> Pj {
        let len = definition.len();
        let mut ptrs: Vec<*mut i8> = Vec::with_capacity(len);
        for s in definition {
            let c_str: std::ffi::CString = std::ffi::CString::new(*s).expect("CString::new failed");
            let ptr = c_str.as_ptr() as *mut i8; // Convert to *mut i8
            ptrs.push(ptr);
        }
        Pj {
            pj: unsafe { proj_sys::proj_create_argv(self.ctx, len as i32, ptrs.as_mut_ptr()) },
        }
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs
    pub fn proj_create_crs_to_crs(&self, source_crs: &str, target_crs: &str, area: PjArea) -> Pj {
        Pj {
            pj: unsafe {
                proj_sys::proj_create_crs_to_crs(
                    self.ctx,
                    source_crs.as_ptr() as *const i8,
                    target_crs.as_ptr() as *const i8,
                    area.area,
                )
            },
        }
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs_from_pj
    pub fn proj_create_crs_to_crs_from_pj(
        &self,
        source_crs: Pj,
        target_crs: Pj,
        area: PjArea,
        authority: Option<&str>,
        accuracy: Option<f64>,
        allow_ballpark: Option<bool>,
        only_best: Option<bool>,
        force_over: Option<bool>,
    ) -> Pj {
        let mut options: Vec<*const i8> = Vec::with_capacity(5);
        if let Some(authority) = authority {
            options.push(format!("AUTHORITY={}", authority).as_ptr() as *mut i8);
        }
        if let Some(accuracy) = accuracy {
            options.push(format!("ACCURACY={}", accuracy).as_ptr() as *mut i8);
        }
        if let Some(allow_ballpark) = allow_ballpark {
            options.push(
                format!(
                    "ALLOW_BALLPARK={}",
                    if allow_ballpark { "YES" } else { "NO" }
                )
                .as_ptr() as *mut i8,
            );
        }
        if let Some(only_best) = only_best {
            options.push(
                format!("ONLY_BEST={}", if only_best { "YES" } else { "NO" }).as_ptr() as *mut i8,
            );
        }
        if let Some(force_over) = force_over {
            options.push(
                format!("FORCE_OVER={}", if force_over { "YES" } else { "NO" }).as_ptr() as *mut i8,
            );
        }
        Pj {
            pj: unsafe {
                proj_sys::proj_create_crs_to_crs_from_pj(
                    self.ctx,
                    source_crs.pj,
                    target_crs.pj,
                    area.area,
                    options.as_ptr(),
                )
            },
        }
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_normalize_for_visualization
    fn _normalize_for_visualization() {
        unimplemented!()
    }
}
impl PjContext {
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_context_errno
    fn _errno(&self) -> PjErrorCode {
        PjErrorCode::from(unsafe { proj_sys::proj_context_errno(self.ctx) } as u32)
    }
    fn _errno_string(&self, err: PjErrorCode) -> String {
        crate::c_char_to_string(unsafe {
            proj_sys::proj_context_errno_string(self.ctx, i32::from(err))
        })
    }
}

pub struct PjArea {
    area: *mut proj_sys::PJ_AREA,
}
impl PjArea {
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_area_create
    pub fn new() -> Self {
        Self {
            area: unsafe { proj_sys::proj_area_create() },
        }
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_area_set_bbox
    pub fn set_bbox(
        &self,
        west_lon_degree: f64,
        south_lat_degree: f64,
        east_lon_degree: f64,
        north_lat_degree: f64,
    ) -> &Self {
        unsafe {
            proj_sys::proj_area_set_bbox(
                self.area,
                west_lon_degree,
                south_lat_degree,
                east_lon_degree,
                north_lat_degree,
            )
        };
        self
    }
}
// region:Info structures
///https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INFO
pub struct PjInfo {
    major: i32,
    minor: i32,
    patch: i32,
    release: String,
    version: String,
    searchpath: String,
}
impl PjInfo {
    pub fn new(
        major: i32,
        minor: i32,
        patch: i32,
        release: String,
        version: String,
        searchpath: String,
    ) -> Self {
        Self {
            major,
            minor,
            patch,
            release,
            version,
            searchpath,
        }
    }
    pub fn major(&self) -> i32 {
        self.major
    }
    pub fn minor(&self) -> i32 {
        self.minor
    }
    pub fn patch(&self) -> i32 {
        self.patch
    }
    pub fn release(&self) -> &str {
        &self.release
    }
    pub fn version(&self) -> &str {
        &self.version
    }
    pub fn searchpath(&self) -> &str {
        &self.searchpath
    }
}
pub struct PjProjInfo {
    id: String,
    description: String,
    definition: String,
    has_inverse: bool,
    accuracy: f64,
}
impl PjProjInfo {
    pub(crate) fn new(
        id: String,
        description: String,
        definition: String,
        has_inverse: bool,
        accuracy: f64,
    ) -> Self {
        Self {
            id,
            description,
            definition,
            has_inverse,
            accuracy,
        }
    }
    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn description(&self) -> &str {
        &self.description
    }
    pub fn definition(&self) -> &str {
        &self.definition
    }
    pub fn has_inverse(&self) -> bool {
        self.has_inverse
    }
    pub fn accuracy(&self) -> f64 {
        self.accuracy
    }
}
pub struct PjGridInfo {
    gridname: String,
    filename: String,
    format: String,
    // lowerleft: String,
    // upperright: String,
    n_lon: i32,
    n_lat: i32,
    cs_lon: f64,
    cs_lat: f64,
}
impl PjGridInfo {
    pub(crate) fn new() -> Self {
        unimplemented!()
    }
    pub fn gridname(&self) -> &str {
        &self.gridname
    }
    pub fn filename(&self) -> &str {
        &self.filename
    }
    pub fn format(&self) -> &str {
        &self.format
    }
    pub fn lowerleft(&self) -> &str {
        unimplemented!()
    }
    pub fn upperright(&self) -> &str {
        unimplemented!()
    }
    pub fn n_lon(&self) -> i32 {
        self.n_lon
    }
    pub fn n_lat(&self) -> i32 {
        self.n_lat
    }
    pub fn cs_lon(&self) -> f64 {
        self.cs_lon
    }
    pub fn cs_lat(&self) -> f64 {
        self.cs_lat
    }
}
pub struct PjInitInfo {
    name: String,
    filename: String,
    version: String,
    origin: String,
    lastupdate: String,
}
impl PjInitInfo {
    pub(crate) fn new(
        name: String,
        filename: String,
        version: String,
        origin: String,
        lastupdate: String,
    ) -> Self {
        Self {
            name,
            filename,
            version,
            origin,
            lastupdate,
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn filename(&self) -> &str {
        &self.filename
    }
    pub fn version(&self) -> &str {
        &self.version
    }
    pub fn origin(&self) -> &str {
        &self.origin
    }
    pub fn lastupdate(&self) -> &str {
        &self.lastupdate
    }
}
// region:Error codes
enum PjErrorCode {
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
