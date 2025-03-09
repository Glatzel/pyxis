use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex};

use cust::prelude::*;
pub(crate) struct PyxisPtx {
    pub name: &'static str,
    pub content: &'static str,
    pub size: usize,
}

/// A struct to manage the currently active module
pub static CONTEXT: LazyLock<PyxisCudaContext> = LazyLock::new(PyxisCudaContext::new);
pub struct PyxisCudaContext {
    _ctx: Context,
    pub(crate) stream: Stream,

    module_cache: Mutex<HashMap<&'static str, (Arc<Module>, usize)>>,
    lru: Mutex<VecDeque<&'static str>>,
    total_size: Mutex<usize>,
    size_limit: Mutex<usize>,
    count_limit: Mutex<usize>,
}

impl PyxisCudaContext {
    /// Create a new module manager

    fn new() -> Self {
        Self {
            _ctx: cust::quick_init().unwrap(),
            stream: Stream::new(StreamFlags::NON_BLOCKING, 1i32.into()).unwrap(),
            module_cache: Mutex::new(HashMap::new()),
            lru: Mutex::new(VecDeque::new()),
            total_size: Mutex::new(0),
            size_limit: Mutex::new(0),
            count_limit: Mutex::new(0),
        }
    }
}
// manage module
impl PyxisCudaContext {
    /// - size_limit: max size of cached ptx file, 0 means unlimited.
    pub fn set_size_limit(&self, size_limit: usize) -> &Self {
        *self.size_limit.lock().unwrap() = size_limit;
        self
    }
    /// - count_limit: max count of cuda modules, 0 means unlimited.
    pub fn set_count_limit(&self, count_limit: usize) -> &Self {
        *self.count_limit.lock().unwrap() = count_limit;
        self
    }
    /// Get the active module or load it if not already loaded
    pub(crate) fn get_module(&self, ptx: &PyxisPtx) -> Arc<Module> {
        // add module to cache if it is not exist in cache
        if !self.module_cache.lock().unwrap().contains_key(ptx.name) {
            self.add_module(ptx);
        }

        // set this module as newest
        self.lru.lock().unwrap().push_front(ptx.name);

        self.module_cache
            .lock()
            .unwrap()
            .get(ptx.name)
            .unwrap()
            .0
            .clone()
    }
    fn add_module(&self, ptx: &PyxisPtx) {
        // clear last module if reach count limit
        if *self.count_limit.lock().unwrap() > 0
            && self.lru.lock().unwrap().len() + 1 > *self.count_limit.lock().unwrap()
        {
            self.remove_last_module();
        }
        // clear modules until total size smaller than limit
        while *self.total_size.lock().unwrap() + ptx.size > *self.size_limit.lock().unwrap() {
            self.remove_last_module();
        }
        // add new module
        self.lru.lock().unwrap().push_front(ptx.name);
        self.module_cache.lock().unwrap().insert(
            ptx.name,
            (
                Arc::new(Module::from_ptx(ptx.content, &[]).unwrap()),
                ptx.size,
            ),
        );
        *self.total_size.lock().unwrap() += ptx.size;
    }
    fn remove_last_module(&self) {
        if let Some(old_key) = self.lru.lock().unwrap().pop_back() {
            if let Some((_, old_size)) = self.module_cache.lock().unwrap().remove(&old_key) {
                *self.total_size.lock().unwrap() -= old_size;
            }
        }
    }
}

// utils
impl PyxisCudaContext {
    pub fn device_buffer_from_slice(&self, slice: &[f64]) -> DeviceBuffer<f64> {
        DeviceBuffer::from_slice(slice).unwrap()
    }
    /// # Returns
    /// (grid_size, block_size)
    pub(crate) fn get_grid_block(&self, func: &Function, length: usize) -> (u32, u32) {
        let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();
        let grid_size = (length as u32).div_ceil(block_size);

        clerk::debug!(
            "using {} blocks and {} threads per block.",
            grid_size,
            block_size
        );

        (grid_size, block_size)
    }
}
