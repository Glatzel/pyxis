use std::ffi::c_void;
use std::ptr::null_mut;

use crate::data_types::LogLevel;

impl Default for crate::Context {
    fn default() -> Self {
        let ctx = Self::new();
        ctx.set_log_level(LogLevel::None).unwrap();
        ctx.set_log_func(null_mut::<c_void>(), Some(crate::proj_clerk))
            .unwrap();
        ctx
    }
}

impl crate::Context {
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_create>
    pub fn new() -> Self {
        Self {
            ptr: unsafe { proj_sys::proj_context_create() },
        }
    }
}

impl Clone for crate::Context {
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_clone>
    fn clone(&self) -> Self {
        Self {
            ptr: unsafe { proj_sys::proj_context_clone(self.ptr) },
        }
    }
}

impl Drop for crate::Context {
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_destroy>
    fn drop(&mut self) { unsafe { proj_sys::proj_context_destroy(self.ptr) }; }
}
