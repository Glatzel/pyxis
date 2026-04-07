use bpaf::{Bpaf, batteries};
pub mod transform;
use bpaf::Parser;
use clerk::LevelFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use transform::transform_args;

#[derive(Clone, Debug, Bpaf)]
#[bpaf(options, version)]
pub struct Args {
    #[bpaf(external(verbose))]
    verbose: LevelFilter,
    #[bpaf(external)]
    sub_commands: SubCommands,
}
#[derive(Bpaf, Clone, Debug)]
pub enum SubCommands {
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
        #[bpaf(
            short,
            long,
            fallback(transform::OutputFormat::Simple),
            display_fallback
        )]
        output_format: transform::OutputFormat,
        #[bpaf(external, many)]
        transform_args: Vec<transform::TransformArgs>,
    },
}

fn verbose() -> impl Parser<LevelFilter> {
    batteries::verbose_by_slice(
        2,
        [
            LevelFilter::OFF,
            LevelFilter::ERROR,
            LevelFilter::WARN,
            LevelFilter::INFO,
            LevelFilter::DEBUG,
            LevelFilter::TRACE,
        ],
    )
}

/// Asynchronous entry point for the CLI tool.
///
/// - Initializes logging and deadlock detection
/// - Loads or overwrites settings
/// - Dispatches subcommand to appropriate handler
pub fn execute() -> mischief::Result<()> {
    // Parse command-line arguments into structured form
    let args = args().run();

    // Initialize logging system with the specified verbosity level
    tracing_subscriber::registry()
        .with(clerk::terminal_layer(true).with_filter(clerk::level_filter(args.verbose)))
        .init();

    // Print parsed arguments at debug level
    tracing::debug!("{:?}", args);

    // Match and execute the selected subcommand
    match args.sub_commands {
        // Run the abacus subcommand with given name and coordinates
        SubCommands::Transform {
            name,
            x,
            y,
            z,
            output_format,
            transform_args,
            ..
        } => transform::execute(&name, x, y, z, output_format, transform_args)?,
    };

    Ok(())
}
