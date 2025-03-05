use cust::prelude::*;
use std::sync::LazyLock;

pub struct PyxisCudaContext {
    _ctx: Context,
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
        &*STREAM
    }
    pub fn from_slice(&self, slice: &[f64]) -> DeviceBuffer<f64> {
        DeviceBuffer::from_slice(slice).unwrap()
    }
}
