use std::ffi::c_void;
use std::ptr::null_mut;
impl Default for crate::PjContext {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::PjContext {
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_create>
    pub fn new() -> Self {
        let ctx = Self {
            ctx: unsafe { proj_sys::proj_context_create() },
        };
        //initialize log
        unsafe {
            proj_sys::proj_log_level(ctx.ctx, i32::from(crate::PjLogLevel::None));
            proj_sys::proj_log_func(ctx.ctx, null_mut::<c_void>(), Some(crate::proj_clerk));
        };
        ctx
    }
}
impl Clone for crate::PjContext {
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_clone>
    fn clone(&self) -> Self {
        Self {
            ctx: unsafe { proj_sys::proj_context_clone(self.ctx) },
        }
    }
}
impl Drop for crate::PjContext {
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_destroy>
    fn drop(&mut self) {
        unsafe { proj_sys::proj_context_destroy(self.ctx) };
    }
}
