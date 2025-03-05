use std::sync::LazyLock;

use cust::prelude::*;

pub struct PyxisCudaContext {
    _ctx: Context,
}
impl Default for PyxisCudaContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PyxisCudaContext {
    pub fn new() -> Self {
        Self {
            _ctx: cust::quick_init().unwrap(),
        }
    }
    pub fn stream(&self) -> &Stream {
        static STREAM: LazyLock<Stream> =
            LazyLock::new(|| Stream::new(StreamFlags::NON_BLOCKING, None).unwrap());
        &STREAM
    }
    pub fn from_slice(&self, slice: &[f64]) -> DeviceBuffer<f64> {
        DeviceBuffer::from_slice(slice).unwrap()
    }
    pub fn size(&self, func: &Function, length: usize) -> (u32, u32) {
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
