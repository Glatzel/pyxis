use std::ffi::CString;
use std::path::Path;

use crate::{check_context_result, check_context_result_inner};

impl crate::PjContext {
    pub fn set_fileapi(&self) {
        unimplemented!()
    }
    pub fn set_sqlite3_vfs_name(&self) {
        unimplemented!()
    }
    pub fn set_file_finder(&self) {
        unimplemented!()
    }
    pub fn set_search_paths(&self, paths: &[&Path]) -> miette::Result<&Self> {
        let len = paths.len();
        let paths: Vec<CString> = paths
            .iter()
            .map(|p| std::ffi::CString::new(p.to_string_lossy().to_string()).unwrap())
            .collect();
        let paths_ptr: Vec<*const i8> = paths.iter().map(|p| p.as_ptr()).collect();
        unsafe {
            proj_sys::proj_context_set_search_paths(self.ctx, len as i32, paths_ptr.as_ptr());
        };
        check_context_result!(self);
        Ok(self)
    }
    pub fn set_ca_bundle_path(&self) {
        unimplemented!()
    }
}
