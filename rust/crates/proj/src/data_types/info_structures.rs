crate::create_readonly_struct!(
    PjInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INFO>",
    {major: i32},
    {minor: i32},
    {patch: i32},
    {release: String},
    {version: String},
    {searchpath: String}
);

crate::create_readonly_struct!(
    PjProjInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_PROJ_INFO>",
    {id: String},
    {description: String},
    {definition: String},
    {has_inverse: bool},
    {accuracy: f64}
);

crate::create_readonly_struct!(
    PjGridInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_GRID_INFO>",
    {gridname: String},
    {filename: String},
    {format: String},
    {lowerleft: crate::PjLp},
    {upperright: crate::PjLp},
    {n_lon: i32},
    {n_lat: i32},
    {cs_lon: f64},
    {cs_lat: f64}
);

crate::create_readonly_struct!(
    PjInitInfo,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_INIT_INFO>",
    {name: String},
    {filename: String},
    {version: String},
    {origin: String},
    {lastupdate: String}
);
