use std::ffi::{CStr, c_char};

use miette::IntoDiagnostic;

pub(crate) fn c_char_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string()
}
pub(crate) fn string_to_c_char(text: &str) -> miette::Result<*const c_char> {
    let c_str: std::ffi::CString = std::ffi::CString::new(text).into_diagnostic()?;
    let ptr = c_str.as_ptr() as *mut i8; // Convert to *mut i8
    Ok(ptr)
}
#[macro_export]
macro_rules! create_readonly_struct {
    ($name:ident, $struct_doc:expr, $({$field:ident: $type:ty $(, $field_doc:expr)?}),*) => {

        #[doc=$struct_doc]
        #[derive(Debug)]
        pub struct $name {
            $( $field: $type ),*
        }

        impl $name {
            // Constructor function to initialize the struct
            pub fn new($($field: $type),*) -> Self {
                $name {
                    $( $field ),*
                }
            }

            // Getter methods for each field
            $(

                pub fn $field(&self) -> &$type {
                    &self.$field
                }
            )*
        }
    }
}
