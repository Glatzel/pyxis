use derive_getters::Getters;

///Description a PROJ operation
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct Operations {
    /// Operation keyword.
    id: String,
    /// Description of operation.
    descr: String,
}
impl Operations {
    pub fn new(id: String, descr: String) -> Self { Operations { id, descr } }
}
///Description of ellipsoids defined in PROJ
///
///# References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct Ellps {
    /// Keyword for the ellipsoid.
    id: String,
    /// Semi-major axis of the ellipsoid, or radius in case of a sphere.
    major: String,
    /// Elliptical parameter, e.g. `rf=298.257` or `b=6356772.2`.
    ell: String,
    /// Name of the ellipsoid
    name: String,
}
impl Ellps {
    pub fn new(id: String, major: String, ell: String, name: String) -> Self {
        Ellps {
            id,
            major,
            ell,
            name,
        }
    }
}
/// Distance units defined in PROJ
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct Units {
    /// Keyword for the unit.
    id: String,
    /// Text representation of the factor that converts a given unit to meters
    to_meter: String,
    /// Name of the unit.
    name: String,
    /// Conversion factor that converts the unit to meters.
    factor: f64,
}
impl Units {
    pub fn new(id: String, to_meter: String, name: String, factor: f64) -> Self {
        Units {
            id,
            to_meter,
            name,
            factor,
        }
    }
}

/// Hard-coded prime meridians defined in PROJ. Note that the structure is no
/// longer updated, and some values may conflict with other sources.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_OPERATIONS>
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Getters)]
pub struct PrimeMeridians {
    /// Keyword for the prime meridian
    id: String,
    /// Offset from Greenwich in DMS format.
    defn: String,
}
impl PrimeMeridians {
    pub fn new(id: String, defn: String) -> Self { PrimeMeridians { id, defn } }
}
