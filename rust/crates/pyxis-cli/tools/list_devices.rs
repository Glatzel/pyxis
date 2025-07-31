use rax::device::{DeviceFilter, list_devices};

fn main() -> miette::Result<()> {
    let devices = list_devices(DeviceFilter::all)?;
    println!("{} devices found.", devices.len());
    for d in devices {
        println!("{d:#?}");
    }
    Ok(())
}
