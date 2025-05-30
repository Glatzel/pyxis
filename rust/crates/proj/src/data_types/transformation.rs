use num_enum::{IntoPrimitive, TryFromPrimitive};

///Object containing everything related to a given projection or
/// transformation. As a user of the PROJ library you are only exposed to
/// pointers to this object and the contents is hidden behind the public API.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ>
pub struct Proj<'a> {
    ptr: *mut proj_sys::PJ,
    pub(crate) ctx: &'a crate::Context,
}
impl Proj<'_> {
    /// Create a `Proj` object from pointer, panic if pointer is null.
    pub(crate) fn new(
        ctx: &crate::Context,
        ptr: *mut proj_sys::PJ,
    ) -> miette::Result<crate::Proj<'_>> {
        if ptr.is_null() {
            miette::bail!("Proj pointer is null.");
        }
        Ok(crate::Proj { ctx, ptr })
    }
    pub(crate) fn ptr(&self) -> *mut proj_sys::PJ { self.ptr }
}
/// Enumeration that is used to convey in which direction a given transformation
/// should be performed. Used in transformation function call as described in
/// the section on transformation functions.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_DIRECTION>
#[derive(Debug, TryFromPrimitive, IntoPrimitive)]
#[repr(i32)]
pub enum Direction {
    ///Perform transformation in the forward direction.
    Fwd = proj_sys::PJ_DIRECTION_PJ_FWD,
    ///Identity. Do nothing.
    Ident = proj_sys::PJ_DIRECTION_PJ_IDENT,
    ///Perform transformation in the inverse direction.
    Inv = proj_sys::PJ_DIRECTION_PJ_INV,
}

///Context objects enable safe multi-threaded usage of PROJ. Each PJ object is
/// connected to a context (if not specified, the default context is used). All
/// operations within a context should be performed in the same thread.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_CONTEXT>
pub struct Context {
    pub(crate) ptr: *mut proj_sys::PJ_CONTEXT,
}
///Opaque object describing an area in which a transformation is performed.
///
///It is used with proj_create_crs_to_crs() to select the best transformation
/// between the two input coordinate reference systems.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_AREA>
pub struct Area {
    pub(crate) ptr: *mut proj_sys::PJ_AREA,
}
