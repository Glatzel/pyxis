use num_enum::{IntoPrimitive, TryFromPrimitive};

#[derive(Debug, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum LogLevel {
    None = 0,
    Error = 1,
    Debug = 2,
    Trace = 3,
    Tell = 4,
}
impl Default for LogLevel {
    fn default() -> Self { Self::Error }
}
