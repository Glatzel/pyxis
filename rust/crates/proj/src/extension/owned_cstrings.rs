extern crate alloc;
use alloc::ffi::CString;
use core::ffi::c_char;
use core::ptr;

use envoy::ToCString;

use crate::data_types::ProjError;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct OwnedCStrings {
    _owned_cstrings: Vec<CString>,
}
impl OwnedCStrings {
    pub fn new() -> Self {
        Self {
            _owned_cstrings: Vec::with_capacity(0),
        }
    }
    pub fn with_capacity(n: usize) -> Self {
        Self {
            _owned_cstrings: Vec::with_capacity(n),
        }
    }
    pub fn push<T: ToCString>(&mut self, value: T) -> Result<*const c_char, ProjError> {
        self._owned_cstrings.push(value.to_cstring()?);
        Ok(self
            ._owned_cstrings
            .last()
            .ok_or(ProjError::new("Last owned cstring is missing.".to_string()))?
            .as_ptr())
    }
    pub fn push_option<T: ToCString>(
        &mut self,
        value: Option<T>,
    ) -> Result<*const c_char, ProjError> {
        match value {
            Some(v) => Ok(self.push(v)?),
            None => Ok(ptr::null()),
        }
    }
    pub fn len(&self) -> usize { self._owned_cstrings.len() }
}
