

///Enum of logging levels in PROJ. Used to set the logging level in PROJ.
/// Usually using [`crate::Context::set_log_level`].
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/datatypes.html#c.PJ_LOG_LEVEL>
#[derive(Debug)]
#[repr(u32)]
pub enum LogLevel {
    ///Don't log anything.
    None = proj_sys::PJ_LOG_LEVEL_PJ_LOG_NONE,
    ///Log only errors.
    Error = proj_sys::PJ_LOG_LEVEL_PJ_LOG_ERROR,
    ///Log errors and additional debug information.
    Debug = proj_sys::PJ_LOG_LEVEL_PJ_LOG_DEBUG,
    ///Highest logging level. Log everything including very detailed debug
    /// information.
    Trace = proj_sys::PJ_LOG_LEVEL_PJ_LOG_TRACE,
    Tell = proj_sys::PJ_LOG_LEVEL_PJ_LOG_TELL,
}
