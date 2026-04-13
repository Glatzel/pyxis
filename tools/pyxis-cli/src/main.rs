pub mod cli;
mod utils;

fn main() -> mischief::Result<()> { crate::cli::execute() }
