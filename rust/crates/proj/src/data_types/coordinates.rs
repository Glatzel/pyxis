// 2D
///Geodetic coordinate, latitude and longitude. Usually in radians.
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_LP>
pub type Lp = proj_sys::PJ_LP;

///2-dimensional cartesian coordinate.
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_XY>
pub type Xy = proj_sys::PJ_XY;

///2-dimensional generic coordinate. Usually used when contents can be either a
/// [`crate::data_types::Xy`] or [`crate::data_types::L`].
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UV>
pub type Uv = proj_sys::PJ_UV;

// 3D
///3-dimensional version of [`crate::data_types::Lp`]. Holds longitude,
/// latitude and a vertical component.
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_LPZ>
pub type Lpz = proj_sys::PJ_LPZ;

///Cartesian coordinate in 3 dimensions. Extension of
/// [`crate::data_types::Xy`].
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_XYZ>
pub type Xyz = proj_sys::PJ_XYZ;

///3-dimensional extension of  [`crate::data_types::Uv`].
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UVW>
pub type Uvw = proj_sys::PJ_UVW;

// Spatiotemporal
///Spatiotemporal version of [`crate::data_types::Lpz`].
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_LPZT>
pub type Lpzt = proj_sys::PJ_LPZT;

///Spatiotemporal version of  [`crate::data_types::Uvw`].
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_XYZT>
pub type Xyzt = proj_sys::PJ_XYZT;
///Geodetic coordinate, latitude and longitude. Usually in radians.
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_UVWT>
pub type Uvwt = proj_sys::PJ_UVWT;

// Ancillary
///Rotations, for instance three euler angles.
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPK>
pub type Opk = proj_sys::PJ_OPK;
///East, north and up components.
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_ENU>
pub type Enu = proj_sys::PJ_ENU;
///Geodesic length, forward and reverse azimuths.
///
/// #references
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GEOD>
pub type Geod = proj_sys::PJ_GEOD;

// Complex
#[cfg(any(feature = "unrecommended", test))]
pub type Coord = proj_sys::PJ_COORD;
