use std::ffi::{c_char, CString};
use std::ptr;

use envoy::ToCString;

#[derive(Debug, Clone)]
pub(crate) struct OwnedCStrings {
    _owned_cstrings: Vec<CString>,
}
impl OwnedCStrings {
    pub fn new() -> Self {
        Self {
            _owned_cstrings: Vec::new(),
        }
    }
    pub fn _with_capacity(n: usize) -> Self {
        Self {
            _owned_cstrings: Vec::with_capacity(n),
        }
    }
    pub fn push<T: ToCString>(&mut self, value: T) -> *const c_char {
        self._owned_cstrings.push(value.to_cstring());
        self._owned_cstrings.last().unwrap().as_ptr()
    }
    pub fn push_option<T: ToCString>(&mut self, value: Option<T>) -> *const c_char {
        match value {
            Some(v) => self.push(v),
            None => ptr::null(),
        }
    }
    pub fn len(&self) -> usize { self._owned_cstrings.len() }
}
