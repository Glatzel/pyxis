use cust::prelude::*;
use std::sync::{Arc, LazyLock, Mutex};
pub static CONTEXT: LazyLock<PyxisCudaContext> = LazyLock::new(|| PyxisCudaContext::new());

/// A struct to manage the currently active module
pub struct PyxisCudaContext {
    _ctx: Context,
    pub stream: Stream,
    active_module: Mutex<Option<Arc<Module>>>,
    active_ptx: Mutex<Option<String>>,
}

impl PyxisCudaContext {
    /// Create a new module manager
    pub fn new() -> Self {
        Self {
            _ctx: cust::quick_init().unwrap(),
            stream: Stream::new(StreamFlags::NON_BLOCKING, 1i32.into()).unwrap(),
            active_module: Mutex::new(None),
            active_ptx: Mutex::new(None),
        }
    }

    /// Get the active module or load it if not already loaded
    pub fn get_module(&self, ptx: &str) -> Result<Arc<Module>, cust::error::CudaError> {
        let mut active_module = self.active_module.lock().unwrap();
        let mut active_ptx = self.active_ptx.lock().unwrap();

        // Check if the requested module is already active
        if let Some(ptx_content) = &*active_ptx {
            if ptx == ptx_content {
                return Ok(active_module.as_ref().unwrap().clone());
            }
        }

        let module = Arc::new(Module::from_ptx(&ptx, &[])?);

        // Update the active module and path
        *active_module = Some(module.clone());
        *active_ptx = Some(ptx.to_string());

        Ok(module)
    }
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
