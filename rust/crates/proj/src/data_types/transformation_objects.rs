use crate::proj_sys;

///Object containing everything related to a given projection or
/// transformation. As a user of the PROJ library you are only exposed to
/// pointers to this object and the contents is hidden behind the public API.
///
/// # References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ>
pub struct Pj {
    pub(crate) pj: *mut proj_sys::PJ,
}

/// Enumeration that is used to convey in which direction a given transformation
/// should be performed. Used in transformation function call as described in
/// the section on transformation functions.
///
/// # References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_DIRECTION>
#[derive(Debug)]
pub enum PjDirection {
    Fwd,
    Ident,
    Inv,
}
impl From<PjDirection> for i32 {
    fn from(value: PjDirection) -> Self {
        match value {
            PjDirection::Fwd => 1,
            PjDirection::Ident => 0,
            PjDirection::Inv => -1,
        }
    }
}
impl From<&PjDirection> for i32 {
    fn from(value: &PjDirection) -> Self {
        match value {
            PjDirection::Fwd => 1,
            PjDirection::Ident => 0,
            PjDirection::Inv => -1,
        }
    }
}
///Context objects enable safe multi-threaded usage of PROJ. Each PJ object is
/// connected to a context (if not specified, the default context is used). All
/// operations within a context should be performed in the same thread.
///
/// # References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_CONTEXT>
pub struct PjContext {
    pub(crate) ctx: *mut proj_sys::PJ_CONTEXT,
}
///Opaque object describing an area in which a transformation is performed.
///
///It is used with proj_create_crs_to_crs() to select the best transformation
/// between the two input coordinate reference systems.
///
/// # References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_AREA>
pub struct PjArea {
    pub(crate) area: *mut proj_sys::PJ_AREA,
}
