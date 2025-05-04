/// # References
///<https://proj.org/en/stable/development/reference/functions.html#c.proj_cleanup>
#[cfg(feature = "unsuggested")]
pub fn cleanup() {
    unsafe { proj_sys::proj_cleanup() };
}
