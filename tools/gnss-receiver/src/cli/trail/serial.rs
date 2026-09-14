use rax::text::Decoder;
use rax_nmea::common::{Identifier, Talker};
use rax_nmea::rules::{NmeaGsvLineCount, NmeaIdentifier, NmeaTalker, NmeaTxtLineCount};
use tokio::io::{AsyncBufReadExt, BufReader};
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
    let mut reader = BufReader::new(serial);
    let mut buf = String::new();

    loop {
        buf.clear();
        match reader.read_line(&mut buf).await {
            Ok(_) => {
                let mut probe = Decoder::new(&buf);
                let talker = probe.global(&NmeaTalker)?;
                let identifier = probe.global(&NmeaIdentifier)?;
                match identifier {
                    Identifier::GSV => {
                        let count = probe.global(&NmeaGsvLineCount)?;
                        for _ in 0..count - 1 {
                            match reader.read_line(&mut buf).await {
                                Ok(_) => {}
                                Err(e) => {
                                    clerk::error!("{e}");
                                    continue;
                                }
                            }
                        }
                    }
                    Identifier::TXT => {
                        let count = probe.global(&NmeaTxtLineCount)?;
                        for _ in 0..count - 1 {
                            match reader.read_line(&mut buf).await {
                                Ok(_) => {}
                                Err(e) => {
                                    clerk::error!("{e}");
                                    continue;
                                }
                            }
                        }
                    }
                    _ => {}
                }
                match tx.send((talker, identifier, buf.clone())).await {
                    Ok(()) => {}
                    Err(e) => {
                        clerk::error!("{e}");
                        continue;
                    }
                }
            }
            Err(e) => {
                clerk::error!("{e}");
                continue;
            }
        }
    }
}
