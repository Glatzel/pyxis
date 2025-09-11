use serialport_ext::{DeviceFilter, list_devices};

fn main() ->mischief::Result<()> {
    let devices = list_devices(DeviceFilter::all)?;
    println!("{} devices found.", devices.len());
    for d in devices {
        println!("{d:#?}");
    }
    Ok(())
}
