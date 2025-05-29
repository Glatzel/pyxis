crate::create_readonly_struct!(
    Info,
    "Struct holding information about the current instance of [`crate::Proj`]. Struct is populated by [`crate::functions::info()`]."
    "# References"
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INFO>",
    {major: i32,"Release info. Version number and release date, e.g. `Rel. 4.9.3, 15 August 2016`."},
    {minor: i32,"Text representation of the full version number, e.g. `4.9.3`."},
    {patch: i32,"Major version number."},
    {release: String,"Minor version number."},
    {version: String,"Patch level of release."},
    {searchpath: String,"Search path for PROJ. List of directories separated by semicolons (Windows) or colons (non-Windows), e.g. `C:\\Users\\doctorwho;C:\\OSGeo4W64\\share\\proj.` Grids and init files are looked for in directories in the search path."}
);

crate::create_readonly_struct!(
    ProjInfo,
     "Struct holding information about a [`crate::Proj`] object. Populated by [`crate::Proj::info()`]. The PJ_PROJ_INFO object provides a view into the internals of a [`crate::Proj`], so once the `Proj` is destroyed or otherwise becomes invalid, so does the [`ProjInfo`]."
     "# References"
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PROJ_INFO>",
    {id: String,"Short ID of the operation the [`crate::Proj`] object is based on, that is, what comes after the `+proj=` in a proj-string, e.g. `merc`."},
    {description: String,"Long describes of the operation the [`crate::Proj`] object is based on, e.g. `Mercator Cyl, Sph&Ell lat_ts=`."},
    {definition: String,"The proj-string that was used to create the [`crate::Proj`] object with, e.g. `+proj=merc +lat_0=24 +lon_0=53 +ellps=WGS84`."},
    {has_inverse: bool},
    {accuracy: f64}
);

crate::create_readonly_struct!(
    GridInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GRID_INFO>",
    {gridname: String},
    {filename: String},
    {format: String},
    {lowerleft: crate::data_types::Lp},
    {upperright: crate::data_types::Lp},
    {n_lon: i32},
    {n_lat: i32},
    {cs_lon: f64},
    {cs_lat: f64}
);

crate::create_readonly_struct!(
    InitInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INIT_INFO>",
    {name: String},
    {filename: String},
    {version: String},
    {origin: String},
    {lastupdate: String}
);
