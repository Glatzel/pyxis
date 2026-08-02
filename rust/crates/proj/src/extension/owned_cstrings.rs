extern crate alloc;
use alloc::ffi::CString;
use core::ffi::c_char;
use core::ptr;

use envoy::ToCString;

use crate::data_types::ProjError;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OwnedCStrings {
    owned_cstrings: Vec<CString>,
}
impl OwnedCStrings {
    pub fn new() -> Self {
        Self {
            owned_cstrings: Vec::with_capacity(0),
        }
    }
    pub fn with_capacity(n: usize) -> Self {
        Self {
            owned_cstrings: Vec::with_capacity(n),
        }
    }
    pub fn push<T>(&mut self, value: T) -> Result<*const c_char, ProjError>
    where
        T: ToCString,
    {
        self.owned_cstrings.push(value.to_cstring()?);
        Ok(self
            .owned_cstrings
            .last()
            .ok_or_else(|| ProjError::new("Last owned cstring is missing.".to_string()))?
            .as_ptr())
    }
    pub fn push_option<T>(&mut self, value: Option<T>) -> Result<*const c_char, ProjError>
    where
        T: ToCString,
    {
        match value {
            Some(v) => Ok(self.push(v)?),
            None => Ok(ptr::null()),
        }
    }
    pub const fn len(&self) -> usize { self.owned_cstrings.len() }
}
