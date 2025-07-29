use miette::{Context, IntoDiagnostic};
use rax::io::{AsyncIRaxReader, AsyncRaxReader};
use rax_nmea::Dispatcher;
use rax_nmea::data::{Identifier, Talker};
use tokio::io::BufReader;
use tokio::sync::mpsc::Sender;
use tokio_serial::SerialPortBuilderExt;
pub fn check_port() -> miette::Result<()> {
    let (port,) = {
        let settings: std::sync::MutexGuard<'_, crate::settings::Settings> = crate::settings::SETTINGS.lock().unwrap();
        (settings.trail_settings.port.clone(),)
    };
    if !tokio_serial::available_ports()
        .into_diagnostic()?
        .iter()
        .any(|p| p.port_name.eq_ignore_ascii_case(&port))
    {
        let msg = format!("Port '{port}' is not available");
        clerk::error!("{msg}");
        if !cfg!(debug_assertions) {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    }
    Ok(())
}
pub async fn start_serial_reader(tx: Sender<(Talker, Identifier, String)>) -> miette::Result<()> {
    let (port, baud_rate) = {
        let settings = crate::settings::SETTINGS.lock().unwrap();
        (
            settings.trail_settings.port.clone(),
            settings.trail_settings.baud_rate,
        )
    };

    let serial = tokio_serial::new(port.clone(), baud_rate)
        .open_native_async()
        .into_diagnostic()
        .wrap_err_with(|| format!("Failed to open serial port: {port}"))?;
    let mut reader = AsyncRaxReader::new(BufReader::new(serial));
    let mut dispatcher = Dispatcher::new();
    loop {
        if let Some(msg) = reader
            .read_line()
            .await?
            .and_then(|l| dispatcher.dispatch(l))
        {
            let _ = tx.send(msg).await;
        }
    }
}
