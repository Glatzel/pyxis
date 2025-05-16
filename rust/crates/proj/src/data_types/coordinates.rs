// region:2D
pub type Lp = proj_sys::PJ_LP;
pub type Xy = proj_sys::PJ_XY;
pub type Uv = proj_sys::PJ_UV;

// region:3D
pub type Lpz = proj_sys::PJ_LPZ;
pub type Xyz = proj_sys::PJ_XYZ;
pub type Uvw = proj_sys::PJ_UVW;

// region:Spatiotemporal
pub type Lpzt = proj_sys::PJ_LPZT;
pub type Xyzt = proj_sys::PJ_XYZT;
pub type Uvwt = proj_sys::PJ_UVWT;

// region:Ancillary
pub type Opk = proj_sys::PJ_OPK;
pub type Enu = proj_sys::PJ_ENU;
pub type Geod = proj_sys::PJ_GEOD;

// region:Complex
#[cfg(any(feature = "unrecommended", test))]
pub type Coord = proj_sys::PJ_COORD;
