crate::create_readonly_struct!(
    Operations,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String},
    // PJ *(*proj)(PJ *);
    {descr: String}
);
crate::create_readonly_struct!(
    Ellps,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String},
    {major: String},
    {ell: String},
    {name: String}
);
crate::create_readonly_struct!(
    Units,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String},
    {to_meter: String},
    {name: String},
    {factor: f64}
);
crate::create_readonly_struct!(
    PrimeMeridians,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String},
    {defn: String}
);
