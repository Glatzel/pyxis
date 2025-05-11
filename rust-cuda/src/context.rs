use std::any::{TypeId, type_name};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock, Mutex};

use cust::prelude::*;
use pyxis::GeoFloat;
pub(crate) struct PyxisPtx {
    pub name: &'static str,
    pub content: &'static str,
    pub size: usize,
}

/// A struct to manage the currently active module
/// default block size 256.
pub static CONTEXT: LazyLock<PyxisCudaContext> = LazyLock::new(PyxisCudaContext::new);
pub struct PyxisCudaContext {
    _ctx: Context,
    pub(crate) stream: Stream,
    block_size: Mutex<u32>,

    module_cache: Mutex<HashMap<&'static str, (Arc<Module>, usize)>>,
    lru: Mutex<VecDeque<&'static str>>,
    total_size: Mutex<usize>,
    size_limit: Mutex<usize>,
    count_limit: Mutex<usize>,
}
/// Create a new module manager
impl PyxisCudaContext {
    fn new() -> Self {
        clerk::debug!("Init the new PyxisCudaContext. size_limit, count_limit are set to 0.");
        Self {
            _ctx: cust::quick_init().unwrap(),
            stream: Stream::new(StreamFlags::NON_BLOCKING, 1i32.into()).unwrap(),
            block_size: Mutex::new(256),

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
        clerk::debug!("Set size_limit to {size_limit}.");
        self
    }
    /// - count_limit: max count of cuda modules, 0 means unlimited.
    pub fn set_count_limit(&self, count_limit: usize) -> &Self {
        *self.count_limit.lock().unwrap() = count_limit;
        clerk::debug!("Set count_limit to {count_limit}.");
        self
    }
    /// Get the active module or load it if not already loaded
    pub(crate) fn get_module(&self, ptx: &PyxisPtx) -> Arc<Module> {
        // add module to cache if it is not exist in cache
        clerk::debug!("Start get module: {}.", ptx.name);
        if !self.module_cache.lock().unwrap().contains_key(ptx.name) {
            clerk::debug!("Module `{}` is not cached.", ptx.name);
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
        clerk::debug!(
            "Start Adding module `{}`, count_limit: {}, size limit: {}, current count: {}, total size: {}",
            ptx.name,
            self.count_limit.lock().unwrap(),
            self.count_limit.lock().unwrap(),
            self.lru.lock().unwrap().len(),
            self.total_size.lock().unwrap()
        );
        if *self.count_limit.lock().unwrap() > 0
            && self.lru.lock().unwrap().len() + 1 > *self.count_limit.lock().unwrap()
        {
            clerk::debug!("Adding module `{}` will exceed count limit", ptx.name);
            self.remove_last_module();
        }
        // clear modules until total size smaller than limit
        while *self.total_size.lock().unwrap() + ptx.size > *self.size_limit.lock().unwrap()
            && !self.lru.lock().unwrap().is_empty()
        {
            clerk::debug!(
                "Adding module `{}` will exceed size limit, total size: {}. Trying to remove last module",
                ptx.name,
                self.total_size.lock().unwrap()
            );
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
                clerk::debug!("Remove last module: `{}`, size: {}`", old_key, old_size);
                clerk::debug!("total_size: `{}`", self.total_size.lock().unwrap());
            }
        }
    }
}
// kernel setting
impl PyxisCudaContext {
    /// - size_limit: max size of cached ptx file, 0 means unlimited.
    pub fn set_block_size(&self, block_size: u32) -> &Self {
        *self.block_size.lock().unwrap() = block_size;
        clerk::debug!("Set size_limit to {block_size}.");
        self
    }
    /// # Returns
    /// (grid_size, block_size) , aka (blocks, threads)
    ///
    /// # References
    /// https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
    pub(crate) fn get_grid_block(&self, length: usize) -> (u32, u32) {
        let grid_size = length as u32 + *self.block_size.lock().unwrap() - 1;
        let grid_size = grid_size.div_ceil(*self.block_size.lock().unwrap());
        println!("{}", grid_size);
        clerk::debug!(
            "using {} blocks and {} threads per block.",
            grid_size,
            self.block_size.lock().unwrap()
        );

        (grid_size, *self.block_size.lock().unwrap())
    }
}
// utils
impl PyxisCudaContext {
    pub(crate) fn get_function<'a, T: 'static + GeoFloat>(
        &self,
        module: &'a Module,
        fn_name: &str,
    ) -> Function<'a> {
        static GEOFLOAT_F32: LazyLock<TypeId> = LazyLock::new(TypeId::of::<f32>);
        static GEOFLOAT_F64: LazyLock<TypeId> = LazyLock::new(TypeId::of::<f64>);
        match TypeId::of::<T>() {
            id if id == *GEOFLOAT_F32 => module.get_function(format!("{fn_name}_float")).unwrap(),
            id if id == *GEOFLOAT_F64 => module.get_function(format!("{fn_name}_double")).unwrap(),
            _ => panic!("Unsupported type: {}", type_name::<T>()),
        }
    }
    pub fn device_buffer_from_slice(&self, slice: &[f64]) -> DeviceBuffer<f64> {
        DeviceBuffer::from_slice(slice).unwrap()
    }
}
