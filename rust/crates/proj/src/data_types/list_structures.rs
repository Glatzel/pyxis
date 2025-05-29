crate::create_readonly_struct!(
    _Operations,
    "Description a PROJ operation"
    "# References"
    "* <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {_id: String,"Operation keyword."},
    // PJ *(*proj)(PJ *);
    {_descr: String,"Description of operation."}
);
crate::create_readonly_struct!(
    Ellps,
    "Description of ellipsoids defined in PROJ"
    "# References"
    "* <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String,"Keyword name of the ellipsoid."},
    {major: String,"Semi-major axis of the ellipsoid, or radius in case of a sphere."},
    {ell: String,"Elliptical parameter, e.g. `rf=298.257` or `b=6356772.2`."},
    {name: String,"Name of the ellipsoid"}
);
crate::create_readonly_struct!(
    Units,
    "Distance units defined in PROJ."
    "# References"
    "* <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String,"Keyword for the unit."},
    {to_meter: String,"Text representation of the factor that converts a given unit to meters"},
    {name: String,"Name of the unit."},
    {factor: f64,"Conversion factor that converts the unit to meters."}
);

crate::create_readonly_struct!(
    PrimeMeridians,
    "Hard-coded prime meridians defined in PROJ. Note that the structure is no longer updated, and some values may conflict with other sources."
    "# References"
    "* <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String,"Keyword for the prime meridian"},
    {defn: String,"Offset from Greenwich in DMS format."}
);
