#![no_std]
#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    unexpected_cfgs,
    dead_code
)]

#[cfg(not(bindgen))]
include!("./bindings.rs");
#[cfg(bindgen)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
