mod cli;
mod logging;
mod settings;
pub use settings::{SETTINGS, Settings};

#[tokio::main]
async fn main() -> mischief::Result<()> { crate::cli::execute().await }
