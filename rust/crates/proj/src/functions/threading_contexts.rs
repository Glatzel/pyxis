impl Default for crate::PjContext {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::PjContext {
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_create>
    pub fn new() -> Self {
        Self {
            ctx: unsafe { proj_sys::proj_context_create() },
        }
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
