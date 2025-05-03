pub struct Pj {
    pub(crate) pj: *mut proj_sys::PJ,
}

/// # References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_DIRECTION>
pub enum PjDirection {
    PjFwd,
    PjIdent,
    PjInv,
}
impl From<PjDirection> for i32 {
    fn from(value: PjDirection) -> Self {
        match value {
            PjDirection::PjFwd => 1,
            PjDirection::PjIdent => 0,
            PjDirection::PjInv => -1,
        }
    }
}
pub struct PjContext {
    pub(crate) ctx: *mut proj_sys::PJ_CONTEXT,
}

pub struct PjArea {
    pub(crate) area: *mut proj_sys::PJ_AREA,
}
