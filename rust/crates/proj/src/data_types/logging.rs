pub enum PjLogLevel {
    None,
    Error,
    Debug,
    Trace,
    Tell,
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
            PjLogLevel::Tell => 4,
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
            PjLogLevel::Tell => 4,
        }
    }
}
impl TryFrom<u32> for PjLogLevel {
    type Error = miette::Report;

    fn try_from(value: u32) -> miette::Result<PjLogLevel> {
        let level = match value {
            0 => PjLogLevel::None,
            1 => PjLogLevel::Error,
            2 => PjLogLevel::Debug,
            3 => PjLogLevel::Trace,
            4 => PjLogLevel::Tell,
            level => miette::bail!("Unknown log level:{}", level),
        };
        Ok(level)
    }
}
