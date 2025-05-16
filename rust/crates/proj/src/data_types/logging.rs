pub enum LogLevel {
    None,
    Error,
    Debug,
    Trace,
    Tell,
}
impl Default for LogLevel {
    fn default() -> Self { Self::Error }
}
impl From<LogLevel> for i32 {
    fn from(value: LogLevel) -> Self {
        match value {
            LogLevel::None => 0,
            LogLevel::Error => 1,
            LogLevel::Debug => 2,
            LogLevel::Trace => 3,
            LogLevel::Tell => 4,
        }
    }
}
impl From<LogLevel> for u32 {
    fn from(value: LogLevel) -> Self {
        match value {
            LogLevel::None => 0,
            LogLevel::Error => 1,
            LogLevel::Debug => 2,
            LogLevel::Trace => 3,
            LogLevel::Tell => 4,
        }
    }
}
impl TryFrom<u32> for LogLevel {
    type Error = miette::Report;

    fn try_from(value: u32) -> miette::Result<LogLevel> {
        let level = match value {
            0 => LogLevel::None,
            1 => LogLevel::Error,
            2 => LogLevel::Debug,
            3 => LogLevel::Trace,
            4 => LogLevel::Tell,
            level => miette::bail!("Unknown log level:{}", level),
        };
        Ok(level)
    }
}
