pub enum PjLogLevel {
    None,
    Error,
    Debug,
    Trace,
}
impl Default for PjLogLevel {
    fn default() -> Self {
        Self::Error
    }
}
impl From<PjLogLevel> for u32 {
    fn from(value: PjLogLevel) -> Self {
        match value {
            PjLogLevel::None => proj_sys::PJ_LOG_LEVEL_PJ_LOG_NONE,
            PjLogLevel::Error => proj_sys::PJ_LOG_LEVEL_PJ_LOG_ERROR,
            PjLogLevel::Debug => proj_sys::PJ_LOG_LEVEL_PJ_LOG_DEBUG,
            PjLogLevel::Trace => proj_sys::PJ_LOG_LEVEL_PJ_LOG_TRACE,
        }
    }
}

