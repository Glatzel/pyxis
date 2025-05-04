#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]
#[cfg(all(not(feature = "bindgen"), target_os = "windows"))]
include!("bindings-win.rs");
#[cfg(all(not(feature = "bindgen"), target_os = "linux"))]
include!("bindings-linux.rs");
#[cfg(feature = "bindgen")]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
