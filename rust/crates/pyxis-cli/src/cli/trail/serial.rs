use rax::io::{AsyncIRaxReader, AsyncRaxReader};
use rax_nmea::Dispatcher;
use rax_nmea::data::{Identifier, Talker};
use tokio::io::BufReader;
use tokio::sync::mpsc::Sender;
use tokio_serial::SerialPortBuilderExt;
pub fn check_port() -> mischief::Result<()> {
    let (port,) = {
        let settings = crate::settings::SETTINGS.lock();
        (settings.trail.port.clone(),)
    };
    if !tokio_serial::available_ports()?
        .iter()
        .any(|p| p.port_name.eq_ignore_ascii_case(&port))
    {
        let msg = format!("Port '{port}' is not available");
        clerk::error!("{msg}");
        if !cfg!(debug_assertions) {
            mischief::bail!("{msg}");
        }
    }
    Ok(())
}
pub async fn start_serial_reader(tx: Sender<(Talker, Identifier, String)>) -> mischief::Result<()> {
    let (port, baud_rate) = {
        let settings = crate::settings::SETTINGS.lock();
        (settings.trail.port.clone(), settings.trail.baud_rate)
    };

    let serial = tokio_serial::new(port.clone(), baud_rate)
        .open_native_async()
        .map_err(|_| mischief::mischief!("Failed to open serial port: {port}"))?;
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
