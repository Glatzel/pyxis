// region:Cleanup
///https://proj.org/en/stable/development/reference/functions.html#c.proj_cleanup
pub fn cleanup() {
    unsafe { proj_sys::proj_cleanup() };
}
// region:C API for ISO-19111 functionality
