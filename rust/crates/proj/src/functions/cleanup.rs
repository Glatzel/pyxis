///This function frees global resources (grids, cache of +init files). It
/// should be called typically before process termination, and after having
/// freed PJ and PJ_CONTEXT objects.
///
/// <div class="warning">Available on <b>crate feature</b>
/// <code>unrecommended</code> only.</div>
///
/// # References
///* <https://proj.org/en/stable/development/reference/functions.html#c.proj_cleanup>
pub fn cleanup() { unsafe { proj_sys::proj_cleanup() }; }
