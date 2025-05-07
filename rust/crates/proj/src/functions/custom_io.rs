use std::ffi::CString;
use std::path::Path;

use miette::IntoDiagnostic;

use crate::check_result;

///Setting custom I/O functions
impl crate::PjContext {
    fn _set_fileapi(&self) { unimplemented!() }

    ///Set the name of a custom SQLite3 VFS.
    ///
    ///This should be a valid SQLite3 VFS name, such as the one passed to the
    /// sqlite3_vfs_register(). See <https://www.sqlite.org/vfs.html>
    ///
    ///It will be used to read proj.db or create&access the cache.db file in
    /// the PROJ user writable directory.
    ///
    ///# References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_sqlite3_vfs_name>
    pub fn set_sqlite3_vfs_name(&self, name: &str) -> miette::Result<&Self> {
        let name = std::ffi::CString::new(name).into_diagnostic()?;
        unsafe {
            proj_sys::proj_context_set_sqlite3_vfs_name(self.ctx, name.as_ptr());
        };
        check_result!(self);
        Ok(self)
    }

    fn _set_file_finder(&self) { unimplemented!() }
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
    ///# References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_search_paths>
    pub fn set_search_paths(&self, paths: &[&Path]) -> miette::Result<&Self> {
        clerk::debug!("search_paths:{:?}", paths);
        let len = paths.len();
        let paths: Vec<CString> = paths
            .iter()
            .map(|p| std::ffi::CString::new(p.to_str().unwrap()).unwrap())
            .collect();
        let paths_ptr: Vec<*const i8> = paths.iter().map(|p| p.as_ptr()).collect();
        unsafe {
            proj_sys::proj_context_set_search_paths(self.ctx, len as i32, paths_ptr.as_ptr());
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
    ///The path should be encoded in UTF-8.
    ///
    ///# References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_ca_bundle_path>
    pub fn set_ca_bundle_path(&self, path: &Path) -> miette::Result<&Self> {
        let path = std::ffi::CString::new(path.to_str().unwrap()).into_diagnostic()?;
        unsafe {
            proj_sys::proj_context_set_ca_bundle_path(self.ctx, path.as_ptr());
        };
        check_result!(self);
        Ok(self)
    }
}
