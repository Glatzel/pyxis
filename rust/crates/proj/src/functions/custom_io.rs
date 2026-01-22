use core::ptr;
use std::path::Path;
extern crate alloc;
use envoy::{AsVecPtr, ToCString, ToVecCString, VecCString};

use crate::check_result;
use crate::data_types::ProjError;

///Setting custom I/O functions
impl crate::Context {
    fn _set_fileapi(&self) { todo!() }

    ///Set the name of a custom SQLite3 VFS.
    ///
    ///This should be a valid SQLite3 VFS name, such as the one passed to the
    /// sqlite3_vfs_register(). See * <https://www.sqlite.org/vfs.html>
    ///
    ///It will be used to read proj.db or create&access the cache.db file in
    /// the PROJ user writable directory.
    ///
    /// # Arguments
    ///
    /// * `name`: SQLite3 VFS name. If NULL is passed, default implementation by
    ///   SQLite will be used.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_sqlite3_vfs_name>
    pub fn set_sqlite3_vfs_name(&self, name: &str) -> Result<&Self, ProjError> {
        unsafe {
            proj_sys::proj_context_set_sqlite3_vfs_name(self.ptr(), name.to_cstring()?.as_ptr());
        };
        check_result!(self);
        Ok(self)
    }
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_file_finder>
    fn _set_file_finder(&self) { todo!() }
    ///Sets search paths.
    ///
    ///Those search paths will be used whenever PROJ must open one of its
    /// resource files (proj.db database, grids, etc...)
    ///
    ///If set on the default context, they will be inherited by contexts
    /// created later.
    ///
    ///Starting with PROJ 7.0, the path(s) should be encoded in UTF-8.
    ///
    /// # Arguments
    ///
    /// * `paths`: Paths. May be empty.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_search_paths>
    pub fn set_search_paths(&self, paths: &[&Path]) -> Result<&Self, ProjError> {
        clerk::debug!("search_paths:{:?}", paths);
        let len = paths.len();
        let paths: VecCString = paths
            .iter()
            .map(|p| p.to_str().unwrap())
            .collect::<Vec<_>>()
            .to_vec_cstring()?;
        unsafe {
            proj_sys::proj_context_set_search_paths(
                self.ptr(),
                len as i32,
                paths.as_vec_ptr().as_ptr(),
            );
        };
        check_result!(self);
        Ok(self)
    }

    ///Sets CA Bundle path.
    ///
    ///Those CA Bundle path will be used by PROJ when curl and PROJ_NETWORK are
    /// enabled.
    ///
    ///If set on the default context, they will be inherited by contexts
    /// created later.
    ///
    /// # Arguments
    ///
    /// * `paths`: Paths. May be None.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_ca_bundle_path>
    pub fn set_ca_bundle_path(&self, path: Option<&Path>) -> Result<&Self, ProjError> {
        let path = path.map(|s| s.to_str().unwrap().to_cstring()).transpose()?;
        unsafe {
            proj_sys::proj_context_set_ca_bundle_path(
                self.ptr(),
                path.map_or(ptr::null(), |p| p.as_ptr()),
            );
        };
        check_result!(self);
        Ok(self)
    }
}
