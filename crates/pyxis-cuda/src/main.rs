use std::error::Error;

use cust::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    let _ctx = cust::quick_init()?;
    let ptx = include_str!("./add.ptx");
    let module = Module::from_ptx(ptx,&[])?;
    const N: usize = 1024;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let func = module.get_function("vector_add")?;

    // Allocate host memory
    let a: Vec<f32> = (0..N).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..N).map(|x| (x * 2) as f32).collect();
    let mut result = vec![0.0f32; N];

    // Allocate device memory
    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let d_result = DeviceBuffer::from_slice(&result)?;

    // Launch kernel (<<<blocks, threads>>>)
    unsafe {
        launch!(
            func<<<5, 256, 0, stream>>>(
                d_a.as_device_ptr(),
                d_b.as_device_ptr(),
                d_result.as_device_ptr(),
                N as i32
            )
        )?;
    }

    // Copy results back to host
    stream.synchronize()?;
    d_result.copy_to(&mut result)?;

    println!("Result: {:?}", &result[..10]); // Print first 10 elements

    Ok(())
}
