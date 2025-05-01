pub struct PjOperations {
    id: String,
    // PJ *(*proj)(PJ *);
    descr: String,
}
pub struct PjEllps {
    id: String,
    major: String,
    ell: String,
    name: String,
}
pub struct PjUnits {
    id: String,
    to_meter: String,
    name: String,
    factor: f64,
}
pub struct PjPrimeMeridians {
    id: String,
    defn: String,
}
