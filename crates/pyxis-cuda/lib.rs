pub fn add() {
    // Allocate host memory
    let a: Vec<f32> = (0..N).map(|x| x as f32).collect();
    let b: Vec<f32> = (0..N).map(|x| (x * 2) as f32).collect();
    let mut result = vec![0.0f32; N];

    // Allocate device memory
    let d_a = DeviceBuffer::from_slice(&a)?;
    let d_b = DeviceBuffer::from_slice(&b)?;
    let mut d_result = DeviceBuffer::from_slice(&result)?;

    // Launch kernel (<<<blocks, threads>>>)
    unsafe {
        launch!(
            func<<<(N + 255) / 256, 256, 0, stream>>>(
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
}
