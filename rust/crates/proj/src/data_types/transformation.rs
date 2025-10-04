extern crate alloc;
use alloc::sync::Arc;


use crate::data_types::ProjError;
use crate::{OwnedCStrings, check_result};

///Object containing everything related to a given projection or
/// transformation. As a user of the PROJ library you are only exposed to
/// pointers to this object and the contents is hidden behind the public API.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ>
pub struct Proj {
    ptr: *mut proj_sys::PJ,
    pub(crate) ctx: Arc<Context>,
    _owned_cstrings: OwnedCStrings,
}
impl Proj {
    /// Create a `Proj` object from pointer, panic if pointer is null.
    pub(crate) fn new(
        ctx: &Arc<Context>,
        ptr: *mut proj_sys::PJ,
    ) -> Result<crate::Proj, ProjError> {
        check_result!(ptr.is_null(), "Proj pointer is null.");
        Ok(crate::Proj {
            ctx: ctx.clone(),
            ptr,
            _owned_cstrings: OwnedCStrings::new(),
        })
    }
    /// Create a `Proj` object from pointer, panic if pointer is null.
    pub(crate) fn new_with_owned_cstrings(
        ctx: &Arc<Context>,
        ptr: *mut proj_sys::PJ,
        owned_cstrings: OwnedCStrings,
    ) -> Result<crate::Proj, ProjError> {
        check_result!(ptr.is_null(), "Proj pointer is null.");
        Ok(crate::Proj {
            ctx: ctx.clone(),
            ptr,
            _owned_cstrings: owned_cstrings,
        })
    }
    pub(crate) fn ptr(&self) -> *mut proj_sys::PJ { self.ptr }
}
/// Enumeration that is used to convey in which direction a given transformation
/// should be performed. Used in transformation function call as described in
/// the section on transformation functions.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_DIRECTION>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug)]
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

unsafe impl Send for Context {}

unsafe impl Sync for Context {}
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

#[cfg(test)]
mod test {
    use super::Proj;

    #[test]
    fn test_proj_new() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = Proj::new(&ctx, std::ptr::null_mut());
        assert!(pj.is_err());
        Ok(())
    }
}
