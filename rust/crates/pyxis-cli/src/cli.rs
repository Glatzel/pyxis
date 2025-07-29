use bpaf::{Bpaf, batteries};
pub mod abacus;
pub mod trail;
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
    Abacus {
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
    #[bpaf(command)]
    Trail {
        /// Serial port to open
        #[bpaf(short, long)]
        port: Option<String>,

        /// Baud rate of the serial port
        #[bpaf(short, long)]
        baud_rate: Option<u32>,

        /// Line buffer capacity
        #[bpaf(short, long)]
        capacity: Option<usize>,
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
async fn execute(cmd: Commands) -> miette::Result<()> {
    //run
    match cmd {
        Commands::Abacus {
            name,
            x,
            y,
            z,
            output_format,
            transform_commands,
        } => abacus::execute(&name, x, y, z, output_format, transform_commands),
        Commands::Trail {
            port,
            baud_rate,
            capacity,
        } => {
            crate::settings::SETTINGS
                .lock()
                .unwrap()
                .overwrite_trail_settings(port, baud_rate, capacity)?;
            trail::execute().await
        }
    }
}
pub async fn cli_main() -> miette::Result<()> {
    let args = args().run();
    crate::logging::init_log(args.verbose);
    tracing::debug!("{:?}", args);
    execute(args.commands).await
}
