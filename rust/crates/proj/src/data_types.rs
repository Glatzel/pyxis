// region:Transformation objects
struct _Proj {}
struct _ProjDirection {}
pub struct ProjContext {
    ctx: *mut proj_sys::PJ_CONTEXT,
}
impl ProjContext {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_create
    pub fn new() -> Self {
        Self {
            ctx: unsafe { proj_sys::proj_context_create() },
        }
    }
}
impl Clone for ProjContext {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_clone
    fn clone(&self) -> Self {
        Self {
            ctx: unsafe { proj_sys::proj_context_clone(self.ctx) },
        }
    }
}
impl Drop for ProjContext {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_destroy
    fn drop(&mut self) {
        unsafe { proj_sys::proj_context_destroy(self.ctx) };
    }
}
struct _ProjArea {}
