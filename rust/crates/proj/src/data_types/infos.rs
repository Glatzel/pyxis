crate::readonly_struct!(
    Info,
    "Struct holding information about the current instance of [`crate::Proj`]. Struct is populated by [`crate::functions::info()`]."
    "# References"
    "* <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INFO>",
    {major: i32,"Release info. Version number and release date, e.g. `Rel. 4.9.3, 15 August 2016`."},
    {minor: i32,"Text representation of the full version number, e.g. `4.9.3`."},
    {patch: i32,"Major version number."},
    {release: String,"Minor version number."},
    {version: String,"Patch level of release."},
    {searchpath: String,"Search path for PROJ. List of directories separated by semicolons (Windows) or colons (non-Windows), e.g. `C:\\Users\\doctorwho;C:\\OSGeo4W64\\share\\proj.` Grids and init files are looked for in directories in the search path."}
);

crate::readonly_struct!(
    ProjInfo,
     "Struct holding information about a [`crate::Proj`] object. Populated by [`crate::Proj::info()`]. The PJ_PROJ_INFO object provides a view into the internals of a [`crate::Proj`], so once the `Proj` is destroyed or otherwise becomes invalid, so does the [`ProjInfo`]."
     "# References"
    "* <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PROJ_INFO>",
    {id: String,"Short ID of the operation the [`crate::Proj`] object is based on, that is, what comes after the `+proj=` in a proj-string, e.g. `merc`."},
    {description: String,"Long describes of the operation the [`crate::Proj`] object is based on, e.g. `Mercator Cyl, Sph&Ell lat_ts=`."},
    {definition: String,"The proj-string that was used to create the [`crate::Proj`] object with, e.g. `+proj=merc +lat_0=24 +lon_0=53 +ellps=WGS84`."},
    {has_inverse: bool,"`true` if an inverse mapping of the defined operation exists, otherwise `false`."},
    {accuracy: f64,"Expected accuracy of the transformation. -1 if unknown."}
);

crate::readonly_struct!(
    GridInfo,
    "Struct holding information about a specific grid in the search path of PROJ. Populated with the function [`crate::functions::grid_info()`]."
    "# References"
    "* <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GRID_INFO>",
    {gridname: String,"Name of grid, e.g. `BETA2007.gsb`."},
    {filename: String,r"Full path of grid file, e.g. `C:\OSGeo4W64\share\proj\BETA2007.gsb`"},
    {format: String,"File format of grid file, e.g. `ntv2`"},
    {lowerleft: crate::data_types::Lp,"Geodetic coordinate of lower left corner of grid."},
    {upperright: crate::data_types::Lp,"Geodetic coordinate of upper right corner of grid."},
    {n_lon: i32,"Number of grid cells in the longitudinal direction."},
    {n_lat: i32,"Number of grid cells in the latitudinal direction."},
    {cs_lon: f64,"Cell size in the longitudinal direction. In radians."},
    {cs_lat: f64,"Cell size in the latitudinal direction. In radians."}
);

crate::readonly_struct!(
    InitInfo,
    "Struct holding information about a specific init file in the search path of PROJ. Populated with the function [`crate::functions::init_info()`]."
     "# References"
    "* <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INIT_INFO>",
    {name: String,"Name of init file, e.g. `epsg`."},
    {filename: String,r"Full path of init file, e.g. `C:\OSGeo4W64\\share\proj\epsg`"},
    {version: String,"Version number of init file, e.g. `9.0.0`"},
    {origin: String,"Originating entity of the init file, e.g. `EPSG`"},
    {lastupdate: String,"Date of last update of the init file."}
);
