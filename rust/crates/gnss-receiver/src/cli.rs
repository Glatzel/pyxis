use bpaf::{Bpaf, batteries};
pub mod trail;
use bpaf::Parser;
use clerk::LogLevel;

#[derive(Clone, Debug, Bpaf)]
#[bpaf(options, version)]
struct Args {
    #[bpaf(external(verbose))]
    verbose: LogLevel,
    #[bpaf(external)]
    sub_commands: SubCommands,
}
#[derive(Bpaf, Clone, Debug)]
pub enum SubCommands {
    #[bpaf(command)]
    Trail {
        /// Serial port to open
        #[bpaf(short, long, fallback(None))]
        port: Option<String>,

        /// Baud rate of the serial port
        #[bpaf(short, long, fallback(None))]
        baud_rate: Option<u32>,

        /// Line buffer capacity
        #[bpaf(short, long, fallback(None))]
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
/// Asynchronous entry point for the CLI tool.
///
/// - Initializes logging and deadlock detection
/// - Loads or overwrites settings
/// - Dispatches subcommand to appropriate handler
pub async fn execute() -> mischief::Result<()> {
    // Parse command-line arguments into structured form
    let args = args().run();

    // Initialize logging system with the specified verbosity level
    crate::logging::init_log(args.verbose);

    // Print parsed arguments at debug level
    tracing::debug!("{:?}", args);

    // Overwrite global SETTINGS with command-line arguments, if applicable
    crate::Settings::overwrite_settings(&args.sub_commands)?;

    // Match and execute the selected subcommand
    match args.sub_commands {
        // Run the interactive TUI trail subcommand
        SubCommands::Trail { .. } => trail::execute().await?,
    };

    Ok(())
}
