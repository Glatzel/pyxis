#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]
#![no_std]

#[cfg(feature = "bindgen")]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
#[cfg(not(feature = "bindgen"))]
extern crate link_cplusplus;
#[cfg(not(feature = "bindgen"))]
mod bindings;
#[cfg(not(feature = "bindgen"))]
pub use bindings::*;
