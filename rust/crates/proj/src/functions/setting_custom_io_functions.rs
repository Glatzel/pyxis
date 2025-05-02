use std::path::Path;

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
        let paths: Vec<*const i8> = paths
            .iter()
            .map(|p| (crate::string_to_c_char(&p.to_string_lossy()).unwrap()) as *const i8)
            .collect();
        unsafe {
            proj_sys::proj_context_set_search_paths(self.ctx, len as i32, paths.as_ptr());
        };
        Ok(self)
    }
    pub fn set_ca_bundle_path(&self) {
        unimplemented!()
    }
}
