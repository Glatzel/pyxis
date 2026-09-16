use core::ffi::{c_char, c_void};

use envoy::PtrToString;

pub unsafe extern "C" fn proj_clerk(_: *mut c_void, level: i32, info: *const c_char) {
    let _message = info.to_string().unwrap_or_default();

    match level {
        1 => {
            clerk::error!("{}", _message);
        }
        2 => {
            clerk::debug!("{}", _message);
        }
        3 => {
            clerk::trace!("{}", _message);
        }
        _ => (),
    }
}
