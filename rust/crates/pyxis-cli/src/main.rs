mod cli;
mod logging;
mod utils;
#[tokio::main]
async fn main() -> miette::Result<()> { cli::cli_main().await }
