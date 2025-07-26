use std::path::PathBuf;
use std::sync::OnceLock;

use chrono::Local;
use clerk::LogLevel;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry;
use tracing_subscriber::util::SubscriberInitExt;

// Global guard to ensure init runs once
static LOG_INIT: OnceLock<()> = OnceLock::new();

pub fn init_log(verbosity: LogLevel) {
    LOG_INIT.get_or_init(|| {
   

        // Generate log file path with datetime
        let log_file_path = generate_log_filename();

        // Create your custom file layer (assumed here as `clerk::file_layer`)
        let file_layer = clerk::file_layer(verbosity, log_file_path, true);

        // Register once
        registry().with(file_layer).init();
    });
}
fn generate_log_filename() -> PathBuf {
    let now = Local::now();
    let filename = format!(
        "log/log-pyxis-trail-{}.log",
        now.format("%Y-%m-%d-%H-%M-%S")
    );

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));

    exe_dir.join(filename)
}