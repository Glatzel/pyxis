use derive_getters::Getters;

///Struct holding information about the current instance of [`crate::Proj`].
/// Struct is populated by [`crate::functions::info()`].
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INFO>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct Info {
    /// Release info. Version number and release date, e.g. `Rel. 4.9.3, 15
    /// August 2016`.
    major: i32,
    /// Text representation of the full version number, e.g. `4.9.3`.
    minor: i32,
    /// Major version number.
    patch: i32,
    /// Minor version number.
    release: String,
    /// Patch level of release.
    version: String,
    /// Search path for PROJ. List of directories separated by semicolons
    /// (Windows) or colons (non-Windows), e.g.
    /// `C:\\Users\\doctorwho;C:\\OSGeo4W64\\share\\proj.` Grids and init files
    /// are looked for in directories in the search path.
    search_path: String,
}
impl Info {
    pub fn new(
        major: i32,
        minor: i32,
        patch: i32,
        release: String,
        version: String,
        search_path: String,
    ) -> Self {
        Info {
            major,
            minor,
            patch,
            release,
            version,
            search_path,
        }
    }
}

///Struct holding information about a [`crate::Proj`] object. Populated by
/// [`crate::Proj::info()`]. The PJ_PROJ_INFO object provides a view into the
/// internals of a [`crate::Proj`], so once the `Proj` is destroyed or otherwise
/// becomes invalid, so does the [`ProjInfo`].
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PROJ_INFO>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct ProjInfo {
    /// Short ID of the operation the [`crate::Proj`] object is based on, that
    /// is, what comes after the `+proj=` in a proj-string, e.g. `merc`.
    id: String,
    /// Long describes of the operation the [`crate::Proj`] object is based on,
    /// e.g. `Mercator Cyl, Sph&Ell lat_ts=`.
    description: String,
    /// The proj-string that was used to create the [`crate::Proj`] object with,
    /// e.g. `+proj=merc +lat_0=24 +lon_0=53 +ellps=WGS84`.
    definition: String,
    /// `true` if an inverse mapping of the defined operation exists, otherwise
    /// `false`.
    has_inverse: bool,
    /// Expected accuracy of the transformation. -1 if unknown.
    accuracy: f64,
}
impl ProjInfo {
    pub fn new(
        id: String,
        description: String,
        definition: String,
        has_inverse: bool,
        accuracy: f64,
    ) -> Self {
        ProjInfo {
            id,
            description,
            definition,
            has_inverse,
            accuracy,
        }
    }
}

///Struct holding information about a specific grid in the search path of PROJ.
/// Populated with the function [`crate::functions::grid_info()`].
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GRID_INFO>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct GridInfo {
    /// Name of grid, e.g. `BETA2007.gsb`.
    gridname: String,
    /// Full path of grid file, e.g. `C:\OSGeo4W64\share\proj\BETA2007.gsb`
    filename: String,
    /// File format of grid file, e.g. `ntv2`
    format: String,
    /// Geodetic coordinate of lower left corner of grid.
    lower_left: (f64, f64),
    /// Geodetic coordinate of upper right corner of grid.
    upper_right: (f64, f64),
    /// Number of grid cells in the longitudinal direction.
    n_lon: i32,
    /// Number of grid cells in the latitudinal direction.
    n_lat: i32,
    /// Cell size in the longitudinal direction. In radians.
    cs_lon: f64,
    /// Cell size in the latitudinal direction. In radians.
    cs_lat: f64,
}
impl GridInfo {
    pub fn new(
        gridname: String,
        filename: String,
        format: String,
        lowerleft: (f64, f64),
        upperright: (f64, f64),
        n_lon: i32,
        n_lat: i32,
        cs_lon: f64,
        cs_lat: f64,
    ) -> Self {
        GridInfo {
            gridname,
            filename,
            format,
            lower_left: lowerleft,
            upper_right: upperright,
            n_lon,
            n_lat,
            cs_lon,
            cs_lat,
        }
    }
}

/// Struct holding information about a specific init file in the search path of
/// PROJ. Populated with the function [`crate::functions::init_info()`]. #
/// References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INIT_INFO>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct InitInfo {
    /// Name of init file, e.g. `epsg`.
    name: String,
    /// Full path of init file, e.g. `C:\OSGeo4W64\share\proj\epsg`
    filename: String,
    /// Version number of init file, e.g. `9.0.0`
    version: String,
    /// Originating entity of the init file, e.g. `EPSG`
    origin: String,
    /// Date of last update of the init file.
    last_update: String,
}
impl InitInfo {
    pub fn new(
        name: String,
        filename: String,
        version: String,
        origin: String,
        lastupdate: String,
    ) -> Self {
        InitInfo {
            name,
            filename,
            version,
            origin,
            last_update: lastupdate,
        }
    }
}
