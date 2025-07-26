mod cli;
mod logging;
mod proj_util;
#[tokio::main]
async fn main() -> miette::Result<()> { cli::cli_main().await }
