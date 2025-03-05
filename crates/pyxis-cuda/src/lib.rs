mod datum_compense_cuda;
use cust::prelude::*;
pub struct PyxisCudaContext {
    _ctx: Context,
}
impl PyxisCudaContext {
    pub fn new() -> Self {
        Self {
            _ctx: cust::quick_init().unwrap(),
        }
    }
    pub fn stream(&self) -> Stream {
        Stream::new(StreamFlags::NON_BLOCKING, None).unwrap()
    }
    pub fn from_slice(&self, slice: &[f64]) -> DeviceBuffer<f64> {
        DeviceBuffer::from_slice(slice).unwrap()
    }
}
