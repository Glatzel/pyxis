pub struct Pj {
    pub(crate) pj: *mut proj_sys::PJ,
}

/// #References
///<https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_DIRECTION>
pub enum PjDirection {
    PjFwd,
    PjIdent,
    PjInv,
}

pub struct PjContext {
    pub(crate) ctx: *mut proj_sys::PJ_CONTEXT,
}

pub struct PjArea {
    pub(crate) area: *mut proj_sys::PJ_AREA,
}
