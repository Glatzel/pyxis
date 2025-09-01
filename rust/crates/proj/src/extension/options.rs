//! Utilities for handling PROJ options and converting Rust types to
//! PROJ-compatible option strings.
//!
//! This module provides:
//! - The `ToProjOptionString` trait for converting types to PROJ option
//!   strings.
//! - A macro for implementing the trait for types that use `to_string()`.
//! - The `ProjOptions` struct for building and managing PROJ options as
//!   CStrings.
extern crate alloc;
use alloc::ffi::CString;
use core::ffi::c_char;

use envoy::ToCString;

/// String constant representing the PROJ option value for `true`.
pub(crate) const OPTION_YES: &str = "YES";
/// String constant representing the PROJ option value for `false`.
pub(crate) const OPTION_NO: &str = "NO";

/// Trait for converting a value to a PROJ-compatible option string.
pub(crate) trait ToProjOptionString {
    /// Converts the value to a string suitable for use as a PROJ option value.
    fn to_option_string(&self) -> String;
}

/// Implements `ToProjOptionString` for `bool`, mapping `true` to `"YES"` and
/// `false` to `"NO"`.
impl ToProjOptionString for bool {
    fn to_option_string(&self) -> String {
        if *self {
            "YES".to_string()
        } else {
            "NO".to_string()
        }
    }
}

/// Macro to implement `ToProjOptionString` for types that can use
/// `to_string()`.
macro_rules! impl_to_option_string {
    ($t:ty) => {
        impl ToProjOptionString for $t {
            fn to_option_string(&self) -> String { self.to_string() }
        }
    };
}

// Use macro for simple types
impl_to_option_string!(f64);
impl_to_option_string!(usize);
impl_to_option_string!(&str);
impl_to_option_string!(crate::data_types::iso19111::AllowIntermediateCrs);

/// Struct for building and managing a list of PROJ options as C-compatible
/// strings.
pub(crate) struct ProjOptions {
    /// The list of options as CStrings, suitable for passing to C APIs.
    options: Vec<CString>,
}

impl ProjOptions {
    /// Creates a new `ProjOptions` with a specified capacity.
    pub fn new(capacity: usize) -> ProjOptions {
        Self {
            options: Vec::with_capacity(capacity),
        }
    }

    /// Pushes a new option with the given name and value, converting the value
    /// using `ToProjOptionString`.
    ///
    /// # Arguments
    /// * `opt` - The value to convert and push.
    /// * `name` - The name of the option.
    pub fn push<T: ToProjOptionString>(&mut self, opt: T, name: &str) -> &mut Self {
        self.options
            .push(format!("{name}={}", opt.to_option_string()).to_cstring());
        self
    }

    /// Pushes an optional value. If `Some`, uses the value; if `None`, uses the
    /// provided default string.
    ///
    /// # Arguments
    /// * `opt` - The optional value to convert and push.
    /// * `name` - The name of the option.
    /// * `default_value` - The string to use if `opt` is `None`.
    pub fn push_optional<T: ToProjOptionString>(
        &mut self,
        opt: Option<T>,
        name: &str,
        default_value: &str,
    ) -> &mut Self {
        match opt {
            Some(opt) => {
                self.options
                    .push(format!("{name}={}", opt.to_option_string()).to_cstring());
            }
            None => {
                self.options
                    .push(format!("{name}={default_value}").to_cstring());
            }
        }
        self
    }

    /// Pushes an optional value. If `Some`, uses the value; if `None`, does
    /// nothing.
    ///
    /// # Arguments
    /// * `opt` - The optional value to convert and push.
    /// * `name` - The name of the option.
    pub fn push_optional_pass<T: ToProjOptionString>(
        &mut self,
        opt: Option<T>,
        name: &str,
    ) -> &mut Self {
        opt.inspect(|o| {
            self.options
                .push(format!("{name}={}", o.to_option_string()).to_cstring());
        });
        self
    }
}
impl envoy::AsVecPtr for ProjOptions {
    fn as_vec_ptr(&self) -> Vec<*const c_char> {
        let mut vec_ptr = self
            .options
            .iter()
            .map(|s| s.as_ptr())
            .collect::<Vec<*const c_char>>();
        vec_ptr.push(std::ptr::null());
        vec_ptr
    }
}
