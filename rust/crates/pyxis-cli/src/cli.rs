use bpaf::{Bpaf, batteries};
mod abacus;
mod trail;
use abacus::transform_commands;
use bpaf::Parser;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Clone, Debug, Bpaf)]
#[bpaf(options, version)]
struct Args {
    #[bpaf(external(verbose))]
    verbose: Level,
    #[bpaf(external)]
    commands: Commands,
}
#[derive(Bpaf, Clone, Debug)]
pub enum Commands {
    #[bpaf(command)]
    Transform {
        #[bpaf(short, long,fallback("".to_string()),)]
        /// Transform task name.
        name: String,
        #[bpaf(short, long, fallback(0.0), display_fallback)]
        /// - X coordinate (in meters).
        ///  - longitude (in degrees).
        ///  - radius of cylindrical (in meters).
        ///  - u of spherical (in radians).
        x: f64,
        #[bpaf(short, long, fallback(0.0), display_fallback)]
        /// - Y coordinate (in meters).
        ///  - latitude (in degrees).
        ///  - u of cylindrical (in radians).
        ///  - v of spherical (in radians).
        y: f64,
        #[bpaf(short, long, fallback(0.0), display_fallback)]
        /// - Z coordinate (in meters).
        ///  - elevation (in meters).
        ///  - z of cylindrical (in meters).
        ///  - radius of spherical (in meters).
        z: f64,
        #[bpaf(short, long, fallback(abacus::OutputFormat::Simple), display_fallback)]
        output_format: abacus::OutputFormat,
        #[bpaf(external, many)]
        transform_commands: Vec<abacus::TransformCommands>,
    },
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
enum Level {
    Quiet,
    Error,
    Warning,
    Info,
    Debug,
    Trace,
}
fn verbose() -> impl Parser<Level> {
    use Level::*;
    batteries::verbose_by_slice(2, [Quiet, Error, Warning, Info, Debug, Trace])
}
fn execute(cmd: Commands) {
    //run
    match cmd {
        Commands::Transform {
            name,
            x,
            y,
            z,
            output_format,
            transform_commands,
        } => abacus::execute(&name, x, y, z, output_format, transform_commands),
    }
}
fn init_log(level: Level) {
    let tracing_level = match level {
        Level::Quiet => clerk::LogLevel::OFF,
        Level::Error => clerk::LogLevel::ERROR,
        Level::Warning => clerk::LogLevel::WARN,
        Level::Info => clerk::LogLevel::INFO,
        Level::Debug => clerk::LogLevel::DEBUG,
        Level::Trace => clerk::LogLevel::TRACE,
    };
    tracing_subscriber::registry()
        .with(clerk::terminal_layer(tracing_level, true))
        .init();
}
pub fn main() {
    let args = args().run();
    init_log(args.verbose);
    tracing::debug!("{:?}", args);
    execute(args.commands);
}
