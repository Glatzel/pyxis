//! Utilities for handling PROJ options and converting Rust types to
//! PROJ-compatible option strings.
//!
//! This module provides:
//! - The `ToProjOptionString` trait for converting types to PROJ option
//!   strings.
//! - A macro for implementing the trait for types that use `to_string()`.
//! - The `ProjOptions` struct for building and managing PROJ options as
//!   CStrings.

use std::ffi::CString;
use std::ptr::null;

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
///
/// # Example
///
/// ```rust
/// impl_to_option_string!(usize);
/// ```
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

/// Struct for building and managing a list of PROJ options as C-compatible
/// strings.
pub(crate) struct ProjOptions {
    /// The list of options as CStrings, suitable for passing to C APIs.
    pub(crate) options: Vec<CString>,
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
    pub fn _push<T: ToProjOptionString>(&mut self, opt: T, name: &str) -> &mut Self {
        self.options.push(
            CString::new(format!("{name}={}", opt.to_option_string()))
                .expect("Error creating CString"),
        );
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
                self.options.push(
                    CString::new(format!("{name}={}", opt.to_option_string()))
                        .expect("Error creating CString"),
                );
            }
            None => {
                self.options.push(
                    CString::new(format!("{name}={default_value}"))
                        .expect("Error creating CString"),
                );
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
        if let Some(opt) = opt {
            self.options.push(
                CString::new(format!("{name}={}", opt.to_option_string()))
                    .expect("Error creating CString"),
            );
        }
        self
    }

    /// Returns a vector of raw pointers to the CStrings, terminated by a null
    /// pointer.
    ///
    /// This is suitable for passing to C APIs that expect a null-terminated
    /// array of strings.
    pub fn vec_ptr(&self) -> Vec<*const i8> {
        let mut ptrs = self.options.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        ptrs.push(null());
        ptrs
    }
}
