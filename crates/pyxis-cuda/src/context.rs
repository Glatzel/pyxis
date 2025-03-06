use core::f64;
use std::sync::LazyLock;

use cust::prelude::*;
pub static CTX: LazyLock<PyxisCudaContext> = LazyLock::new(|| PyxisCudaContext {
    _ctx: cust::quick_init().unwrap(),
    stream: LazyLock::new(|| Stream::new(StreamFlags::NON_BLOCKING, 1i32.into()).unwrap()),
});

pub struct PyxisCudaContext {
    _ctx: Context,
    pub stream: LazyLock<Stream>,
}

impl PyxisCudaContext {
    pub fn from_slice(&self, slice: &[f64]) -> DeviceBuffer<f64> {
        DeviceBuffer::from_slice(slice).unwrap()
    }
    /// # Returns
    /// (grid_size, block_size)
    pub fn get_grid_block(&self, func: &Function, length: usize) -> (u32, u32) {
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();
        let grid_size = (length as u32).div_ceil(block_size);
        #[cfg(feature = "log")]
        {
            tracing::debug!(
                "using {} blocks and {} threads per block.",
                grid_size,
                block_size
            );
        }
        (grid_size, block_size)
    }
}
