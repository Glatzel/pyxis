use clap::Parser;

/// Command-line arguments that override `pyxis-trail.toml`.
#[derive(Debug, Parser)]
pub struct CliArgs {
    /// Serial port to open
    #[arg(short, long)]
    pub port: Option<String>,

    /// Baud rate of the serial port
    #[arg(short, long)]
    pub baud_rate: Option<u32>,

    /// Line buffer capacity
    #[arg(short, long)]
    pub capacity: Option<usize>,
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_default_cli_args() {
        let args = CliArgs::parse_from(["pyxis-trail"]);
        assert_eq!(args.port, None);
        assert_eq!(args.baud_rate, None);
        assert_eq!(args.capacity, None);
    }

    #[test]
    fn test_cli_parsing_port_and_baud_rate() {
        let args = CliArgs::parse_from(["pyxis-trail", "--port", "COM3", "--baud-rate", "115200"]);
        assert_eq!(args.port.as_deref(), Some("COM3"));
        assert_eq!(args.baud_rate, Some(115200));
    }

    #[test]
    fn test_cli_parsing_with_short_flags() {
        let args = CliArgs::parse_from(["pyxis-trail", "-p", "COM9", "-b", "38400", "-c", "512"]);
        assert_eq!(args.port.as_deref(), Some("COM9"));
        assert_eq!(args.baud_rate, Some(38400));
        assert_eq!(args.capacity, Some(512));
    }
}
