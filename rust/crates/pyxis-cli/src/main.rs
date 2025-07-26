mod cli;
mod logging;
mod proj_util;
#[tokio::main]
async fn main() { cli::cli_main(); }
