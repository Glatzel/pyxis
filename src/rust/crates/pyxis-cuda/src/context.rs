use std::cell::{Ref, RefCell};
use std::collections::{HashMap, VecDeque};

use cust::prelude::*;

pub(crate) struct PyxisPtx {
    pub name: &'static str,
    pub content: &'static str,
    pub size: usize,
}

/// A struct to manage the currently active module
pub struct PyxisCudaContext {
    _ctx: Context,
    pub stream: Stream,

    module_cache: RefCell<HashMap<&'static str, (Module, usize)>>,
    lru: RefCell<VecDeque<&'static str>>,
    total_size: RefCell<usize>,
    size_limit: usize,
    count_limit: usize,
}

impl Default for PyxisCudaContext {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl PyxisCudaContext {
    /// Create a new module manager
    /// # Arguments
    /// - size_limit: max size of cached ptx file, 0 means unlimited.
    /// - count_limit: max count of cuda modules, 0 means unlimited.
    pub fn new(size_limit: usize, count_limit: usize) -> Self {
        Self {
            _ctx: cust::quick_init().unwrap(),
            stream: Stream::new(StreamFlags::NON_BLOCKING, 1i32.into()).unwrap(),
            module_cache: RefCell::new(HashMap::new()),
            lru: RefCell::new(VecDeque::new()),
            total_size: RefCell::new(0),
            size_limit,
            count_limit,
        }
    }
}
// manage module
impl PyxisCudaContext {
    /// Get the active module or load it if not already loaded
    pub(crate) fn get_module(&self, ptx: &PyxisPtx) -> Ref<Module> {
        // add module to cache if it is not exist in cache
        if !self.module_cache.borrow().contains_key(ptx.name) {
            self.add_module(ptx);
        }

        // set this module as newest
        self.lru.borrow_mut().push_front(ptx.name);
        Ref::map(self.module_cache.borrow(), |m| &m.get(ptx.name).unwrap().0)
    }
    fn add_module(&self, ptx: &PyxisPtx) {
        // clear last module if reach count limit
        if self.count_limit > 0 && self.lru.borrow().len() + 1 > self.count_limit {
            self.remove_last_module();
        }
        // clear modules until total size smaller than limit
        while *self.total_size.borrow() + ptx.size > self.size_limit {
            self.remove_last_module();
        }
        // add new module
        self.lru.borrow_mut().push_front(ptx.name);
        self.module_cache.borrow_mut().insert(
            ptx.name,
            (Module::from_ptx(ptx.content, &[]).unwrap(), ptx.size),
        );
        *self.total_size.borrow_mut() += ptx.size;
    }
    fn remove_last_module(&self) {
        if let Some(old_key) = self.lru.borrow_mut().pop_back() {
            if let Some((_, old_size)) = self.module_cache.borrow_mut().remove(&old_key) {
                *self.total_size.borrow_mut() -= old_size;
            }
        }
    }
}

// utils
impl PyxisCudaContext {
    pub fn from_slice(&self, slice: &[f64]) -> DeviceBuffer<f64> {
        DeviceBuffer::from_slice(slice).unwrap()
    }
    /// # Returns
    /// (grid_size, block_size)
    pub(crate) fn get_grid_block(&self, func: &Function, length: usize) -> (u32, u32) {
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
