crate::create_readonly_struct!(
    PjOperations,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String},
    // PJ *(*proj)(PJ *);
    {descr: String}
);
crate::create_readonly_struct!(
    PjEllps,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String},
    {major: String},
    {ell: String},
    {name: String}
);
crate::create_readonly_struct!(
    PjUnits,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String},
    {to_meter: String},
    {name: String},
    {factor: f64}
);
crate::create_readonly_struct!(
    PjPrimeMeridians,
    "<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>",
    {id: String},
    {defn: String}
);
