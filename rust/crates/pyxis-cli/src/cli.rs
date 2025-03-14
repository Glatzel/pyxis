use bpaf::{Bpaf, batteries};
mod transform;
use bpaf::Parser;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use transform::transform_commands;

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
        ///
        x: f64,
        #[bpaf(short, long, fallback(0.0), display_fallback)]
        /// - Y coordinate (in meters).
        ///  - latitude (in degrees).
        ///  - u of cylindrical (in radians).
        ///  - v of spherical (in radians).
        ///
        y: f64,
        #[bpaf(short, long, fallback(0.0), display_fallback)]
        /// - Z coordinate (in meters).
        ///  - elevation (in meters).
        ///  - z of cylindrical (in meters).
        ///  - radius of spherical (in meters).
        ///
        z: f64,
        #[bpaf(
            short,
            long,
            fallback(transform::OutputFormat::Simple),
            display_fallback
        )]
        output_format: transform::OutputFormat,
        #[bpaf(external, many)]
        transform_commands: Vec<transform::TransformCommands>,
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
        } => transform::execute(&name, x, y, z, output_format, transform_commands),
    }
}
fn init_log(level: Level) {
    let tracing_level = match level {
        Level::Quiet => tracing::level_filters::LevelFilter::OFF,
        Level::Error => tracing::level_filters::LevelFilter::ERROR,
        Level::Warning => tracing::level_filters::LevelFilter::WARN,
        Level::Info => tracing::level_filters::LevelFilter::INFO,
        Level::Debug => tracing::level_filters::LevelFilter::DEBUG,
        Level::Trace => tracing::level_filters::LevelFilter::TRACE,
    };
    tracing_subscriber::registry()
        .with(clerk::terminal_layer(tracing_level))
        .init();
}
pub fn main() {
    let args = args().run();
    init_log(args.verbose);
    tracing::debug!("{:?}", args);
    execute(args.commands);
}
