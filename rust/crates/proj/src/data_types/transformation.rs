extern crate alloc;
use std::ffi::c_void;
use std::ptr::null_mut;

use alloc::sync::Arc;

use crate::data_types::ProjError;
use crate::{LogLevel, OwnedCStrings, check_result};

///Object containing everything related to a given projection or
/// transformation. As a user of the PROJ library you are only exposed to
/// pointers to this object and the contents is hidden behind the public API.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Proj {
    ptr: *mut proj_sys::PJ,
    arc_ctx_ptr: Arc<ContextPtr>,
    _owned_cstrings: OwnedCStrings,
}
impl Proj {
    /// Create a `Proj` object from pointer, panic if pointer is null.
    pub(crate) fn new(
        arc_ctx_ptr: Arc<ContextPtr>,
        ptr: *mut proj_sys::PJ,
    ) -> Result<crate::Proj, ProjError> {
        check_result!(ptr.is_null(), "Proj pointer is null.");
        Ok(crate::Proj {
            arc_ctx_ptr,
            ptr,
            _owned_cstrings: OwnedCStrings::new(),
        })
    }
    /// Create a `Proj` object from pointer, panic if pointer is null.
    pub(crate) fn new_with_owned_cstrings(
        arc_ctx_ptr: Arc<ContextPtr>,
        ptr: *mut proj_sys::PJ,
        owned_cstrings: OwnedCStrings,
    ) -> Result<crate::Proj, ProjError> {
        check_result!(ptr.is_null(), "Proj pointer is null.");
        Ok(crate::Proj {
            arc_ctx_ptr,
            ptr,
            _owned_cstrings: owned_cstrings,
        })
    }
    pub(crate) fn ptr(&self) -> *mut proj_sys::PJ { self.ptr }
    pub(crate) fn ctx_ptr(&self) -> *mut proj_sys::PJ_CONTEXT { self.arc_ctx_ptr.ptr() }
    pub(crate) fn arc_ctx_ptr(&self) -> Arc<ContextPtr> { self.arc_ctx_ptr.clone() }
}
/// Enumeration that is used to convey in which direction a given transformation
/// should be performed. Used in transformation function call as described in
/// the section on transformation functions.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_DIRECTION>
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(i32)]
pub enum Direction {
    ///Perform transformation in the forward direction.
    Fwd = proj_sys::PJ_DIRECTION_PJ_FWD,
    ///Identity. Do nothing.
    Ident = proj_sys::PJ_DIRECTION_PJ_IDENT,
    ///Perform transformation in the inverse direction.
    Inv = proj_sys::PJ_DIRECTION_PJ_INV,
}
#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub(crate) struct ContextPtr(pub(crate) *mut proj_sys::PJ_CONTEXT);
impl ContextPtr {
    pub(crate) fn ptr(&self) -> *mut proj_sys::PJ_CONTEXT { self.0 }
}
impl Drop for ContextPtr {
    ///Deallocate a threading-context.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_destroy>
    fn drop(&mut self) { unsafe { proj_sys::proj_context_destroy(self.0) }; }
}

///Context objects enable safe multi-threaded usage of PROJ. Each PJ object is
/// connected to a context (if not specified, the default context is used). All
/// operations within a context should be performed in the same thread.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_CONTEXT>
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Context {
    ptr: Arc<ContextPtr>,
}
impl Context {
    pub(crate) fn arc_ptr(&self) -> Arc<ContextPtr> { self.ptr.clone() }
    pub(crate) fn ptr(&self) -> *mut proj_sys::PJ_CONTEXT { self.ptr.0 }
}
unsafe impl Send for Context {}

unsafe impl Sync for Context {}
impl crate::Context {
    ///Create a new threading-context.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_create>
    pub fn new() -> Self {
        let ctx = Self {
            ptr: Arc::new(ContextPtr(unsafe { proj_sys::proj_context_create() })),
        };
        ctx.set_log_level(LogLevel::None).unwrap();
        ctx.set_log_fn(null_mut::<c_void>(), Some(crate::proj_clerk))
            .unwrap();
        ctx
    }
}

impl Clone for crate::Context {
    ///Create a new threading-context based on an existing context.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_clone>
    fn clone(&self) -> Self {
        Self {
            ptr: self.arc_ptr(),
        }
    }
}

///Opaque object describing an area in which a transformation is performed.
///
///It is used with proj_create_crs_to_crs() to select the best transformation
/// between the two input coordinate reference systems.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_AREA>
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Area {
    pub(crate) ptr: *mut proj_sys::PJ_AREA,
}

#[cfg(test)]
mod test {
    use super::Proj;

    #[test]
    fn test_proj_new() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = Proj::new(ctx.ptr.clone(), std::ptr::null_mut());
        assert!(pj.is_err());
        Ok(())
    }
    #[test]
    fn test_clone() -> mischief::Result<()> {
        let ctx1 = crate::new_test_ctx()?;
        let ctx2 = ctx1.clone();
        assert!(!ctx2.ptr().is_null());
        Ok(())
    }
}
