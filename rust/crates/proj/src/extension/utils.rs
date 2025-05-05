use std::ffi::{CStr, c_char};

pub(crate) fn c_char_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }.to_string_lossy().to_string()
}

pub(crate) fn array4_to_pj_coord(array4: [f64; 4]) -> miette::Result<proj_sys::PJ_COORD> {
    let coord = match (array4[2].is_nan(), array4[3].is_nan()) {
        (true, true) => proj_sys::PJ_COORD {
            xy: proj_sys::PJ_XY {
                x: array4[0],
                y: array4[1],
            },
        },
        (false, true) => proj_sys::PJ_COORD {
            xyz: proj_sys::PJ_XYZ {
                x: array4[0],
                y: array4[1],
                z: array4[2],
            },
        },
        (false, false) => proj_sys::PJ_COORD { v: array4 },
        (true, false) => {
            miette::bail!("Component3 is NAN, Component4 is not NAN")
        }
    };
    Ok(coord)
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
