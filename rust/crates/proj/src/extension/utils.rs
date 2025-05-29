use std::ffi::{CStr, c_char};

pub(crate) fn c_char_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string())
}
pub(crate) fn vec_c_char_to_string(ptr: *mut *mut i8) -> Option<Vec<String>> {
    if ptr.is_null() {
        return None;
    }
    let mut vec_str = Vec::new();
    let mut offset = 0;

    loop {
        let current_ptr = unsafe { ptr.offset(offset).as_ref().unwrap() };
        if current_ptr.is_null() {
            break;
        }
        vec_str.push(c_char_to_string(current_ptr.cast_const()).unwrap());
        offset += 1;
    }
    Some(vec_str)
}
macro_rules! create_readonly_struct {
    ($name:ident, $($struct_doc:expr)+, $({$field:ident: $type:ty $(, $field_doc:expr)?}),*) => {
        $(#[doc=$struct_doc])+
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
                $(#[doc=$field_doc])?
                pub fn $field(&self) -> &$type {
                    &self.$field
                }
            )*
        }
    }
}
pub(crate) use create_readonly_struct;
