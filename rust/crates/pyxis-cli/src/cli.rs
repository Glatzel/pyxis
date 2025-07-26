use bpaf::{Bpaf, batteries};
mod abacus;
mod trail;
use abacus::transform_commands;
use bpaf::Parser;
use clerk::LogLevel;

#[derive(Clone, Debug, Bpaf)]
#[bpaf(options, version)]
struct Args {
    #[bpaf(external(verbose))]
    verbose: LogLevel,
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

fn verbose() -> impl Parser<LogLevel> {
    batteries::verbose_by_slice(
        2,
        [
            LogLevel::OFF,
            LogLevel::ERROR,
            LogLevel::WARN,
            LogLevel::INFO,
            LogLevel::DEBUG,
            LogLevel::TRACE,
        ],
    )
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
pub fn cli_main() {
    let args = args().run();
    crate::logging::init_log(args.verbose);
    tracing::debug!("{:?}", args);
    execute(args.commands);
}
