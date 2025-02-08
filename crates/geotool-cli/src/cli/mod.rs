use bpaf::batteries;
use bpaf::Bpaf;
mod transform;
use crate::config_logger;
use bpaf::Parser;
use transform::transform_commands;
#[derive(Clone, Debug, Bpaf)]
#[bpaf(options, version, fallback_to_usage)]
struct Args {
    #[bpaf(external(verbose))]
    verbose: Level,
    #[bpaf(external)]
    commands: Commands,
}
#[derive(Bpaf, Clone, Debug)]
pub enum Commands {
    #[bpaf(command, fallback_to_usage)]
    Transform {
        #[bpaf(short, long)]
        /// - X coordinate (in meters).
        ///  - longitude (in degrees).
        ///  - radius of cylindrical (in meters).
        ///  - u of spherical (in radians).
        x: f64,
        #[bpaf(short, long)]
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
    batteries::verbose_by_slice(1, [Quiet, Error, Warning, Info, Debug, Trace])
}
pub fn main() {
    let args = args().run();
    //config logger
    let log_level = match args.verbose {
        Level::Quiet => tracing::level_filters::LevelFilter::OFF,
        Level::Error => tracing::level_filters::LevelFilter::ERROR,
        Level::Warning => tracing::level_filters::LevelFilter::WARN,
        Level::Info => tracing::level_filters::LevelFilter::INFO,
        Level::Debug => tracing::level_filters::LevelFilter::DEBUG,
        Level::Trace => tracing::level_filters::LevelFilter::TRACE,
    };
    config_logger::init_logger(log_level);
    //run
    match args.commands {
        Commands::Transform {
            x,
            y,
            z,
            output_format,
            transform_commands,
        } => transform::execute(x, y, z, output_format, transform_commands),
    }
}
