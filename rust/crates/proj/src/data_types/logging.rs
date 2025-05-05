pub enum PjLogLevel {
    None,
    Error,
    Debug,
    Trace,
}
impl Default for PjLogLevel {
    fn default() -> Self { Self::Error }
}
impl From<PjLogLevel> for i32 {
    fn from(value: PjLogLevel) -> Self {
        match value {
            PjLogLevel::None => 0,
            PjLogLevel::Error => 1,
            PjLogLevel::Debug => 2,
            PjLogLevel::Trace => 3,
        }
    }
}
impl From<PjLogLevel> for u32 {
    fn from(value: PjLogLevel) -> Self {
        match value {
            PjLogLevel::None => 0,
            PjLogLevel::Error => 1,
            PjLogLevel::Debug => 2,
            PjLogLevel::Trace => 3,
        }
    }
}
