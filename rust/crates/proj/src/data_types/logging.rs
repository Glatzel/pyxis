use num_enum::{IntoPrimitive, TryFromPrimitive};

///Enum of logging levels in PROJ. Used to set the logging level in PROJ.
/// Usually using [`crate::Context::set_log_level`].
///
/// # References
///
/// <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_LOG_LEVEL>
#[derive(Debug, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum LogLevel {
    None = proj_sys::PJ_LOG_LEVEL_PJ_LOG_NONE,
    Error = proj_sys::PJ_LOG_LEVEL_PJ_LOG_ERROR,
    Debug = proj_sys::PJ_LOG_LEVEL_PJ_LOG_DEBUG,
    Trace = proj_sys::PJ_LOG_LEVEL_PJ_LOG_TRACE,
    Tell = proj_sys::PJ_LOG_LEVEL_PJ_LOG_TELL,
}
