mod cli;
mod logging;
mod settings;
mod utils;
pub use settings::{SETTINGS, Settings};

#[tokio::main]
async fn main() ->Result<(),ProjError> { crate::cli::execute().await }
