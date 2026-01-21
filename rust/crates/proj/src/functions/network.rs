use std::path::PathBuf;

use envoy::{CStrToString, ToCString};

use crate::check_result;
use crate::data_types::ProjError;
impl crate::Context {
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_network_callbacks>
    fn _set_network_callbacks(&self) { todo!() }
    ///Enable or disable network access.
    ///
    ///This overrides the default endpoint in the PROJ configuration file or
    /// with the PROJ_NETWORK environment variable.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_enable_network>
    pub fn set_enable_network(&self, enabled: bool) -> Result<&Self, ProjError> {
        let result =
            unsafe { proj_sys::proj_context_set_enable_network(*self.ptr, enabled as i32) } != 0;
        check_result!(enabled ^ result, "Network interface is not available.");
        check_result!(self);
        Ok(self)
    }
    ///Return if network access is enabled.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_is_network_enabled>
    pub fn is_network_enabled(&self) -> Result<bool, ProjError> {
        let result = unsafe { proj_sys::proj_context_is_network_enabled(*self.ptr) } != 0;
        check_result!(self);
        Ok(result)
    }
    ///Define the URL endpoint to query for remote grids.
    ///
    ///This overrides the default endpoint in the PROJ configuration file or
    /// with the PROJ_NETWORK_ENDPOINT environment variable.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_url_endpoint>
    pub fn set_url_endpoint(&self, url: &str) -> Result<&Self, ProjError> {
        unsafe {
            proj_sys::proj_context_set_url_endpoint(*self.ptr, url.to_cstring().as_ptr());
        };
        check_result!(self);
        Ok(self)
    }
    ///Get the URL endpoint to query for remote grids.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_url_endpoint>
    pub fn get_url_endpoint(&self) -> Result<String, ProjError> {
        let result = unsafe { proj_sys::proj_context_get_url_endpoint(*self.ptr) };
        check_result!(self);
        Ok(result.to_string().unwrap_or_default())
    }
    ///Get the PROJ user writable directory for downloadable resource files,
    /// such as datum shift grids.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_user_writable_directory>
    pub fn get_user_writable_directory(&self, create: bool) -> Result<PathBuf, ProjError> {
        let result =
            unsafe { proj_sys::proj_context_get_user_writable_directory(*self.ptr, create as i32) };
        check_result!(self);
        Ok(PathBuf::from(result.to_string().unwrap_or_default()))
    }
    ///Enable or disable the local cache of grid chunks
    ///
    ///This overrides the setting in the PROJ configuration file.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_enable>
    pub fn grid_cache_set_enable(&self, enabled: bool) -> Result<&Self, ProjError> {
        unsafe { proj_sys::proj_grid_cache_set_enable(*self.ptr, enabled as i32) };
        check_result!(self);
        Ok(self)
    }
    ///Override, for the considered context, the path and file of the local
    /// cache of grid chunks.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_filename>
    pub fn grid_cache_set_filename(&self, fullname: &str) -> Result<&Self, ProjError> {
        unsafe {
            proj_sys::proj_grid_cache_set_filename(*self.ptr, fullname.to_cstring().as_ptr())
        };
        check_result!(self);
        Ok(self)
    }
    ///Override, for the considered context, the maximum size of the local
    /// cache of grid chunks.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_max_size>
    pub fn grid_cache_set_max_size(&self, max_size_mbyte: u16) -> Result<&Self, ProjError> {
        unsafe { proj_sys::proj_grid_cache_set_max_size(*self.ptr, max_size_mbyte as i32) };
        check_result!(self);
        Ok(self)
    }
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_ttl>
    pub fn grid_cache_set_ttl(&self, ttl_seconds: u16) -> Result<&Self, ProjError> {
        unsafe { proj_sys::proj_grid_cache_set_ttl(*self.ptr, ttl_seconds as i32) };
        check_result!(self);
        Ok(self)
    }
    ///Clear the local cache of grid chunks.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_clear>
    pub fn grid_cache_clear(&self) -> Result<&Self, ProjError> {
        unsafe { proj_sys::proj_grid_cache_clear(*self.ptr) };
        Ok(self)
    }
    ///Return if a file must be downloaded or is already available in the PROJ
    /// user-writable directory.
    ///
    ///The file will be determinted to have to be downloaded if it does not
    /// exist yet in the user-writable directory, or if it is determined that a
    /// more recent version exists. To determine if a more recent version
    /// exists, PROJ will use the "downloaded_file_properties" table of its grid
    /// cache database. Consequently files manually placed in the user-writable
    /// directory without using this function would be considered as
    /// non-existing/obsolete and would be unconditionally downloaded again.
    ///
    ///This function can only be used if networking is enabled, and either the
    /// default curl network API or a custom one have been installed.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_download_needed>
    pub fn is_download_needed(
        &self,
        url_or_filename: &str,
        ignore_ttl_setting: bool,
    ) -> Result<bool, ProjError> {
        let result = unsafe {
            proj_sys::proj_is_download_needed(
                *self.ptr,
                url_or_filename.to_cstring().as_ptr(),
                ignore_ttl_setting as i32,
            )
        } != 0;
        check_result!(self);
        Ok(result)
    }
    ///Download a file in the PROJ user-writable directory.
    ///
    ///The file will only be downloaded if it does not exist yet in the
    /// user-writable directory, or if it is determined that a more recent
    /// version exists. To determine if a more recent version exists, PROJ will
    /// use the "downloaded_file_properties" table of its grid cache database.
    /// Consequently files manually placed in the user-writable directory
    /// without using this function would be considered as non-existing/obsolete
    /// and would be unconditionally downloaded again.
    ///
    ///This function can only be used if networking is enabled, and either the
    /// default curl network API or a custom one have been installed.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_download_file>
    pub fn download_file(
        &self,
        url_or_filename: &str,
        ignore_ttl_setting: bool,
    ) -> Result<bool, ProjError> {
        let result = unsafe {
            proj_sys::proj_download_file(
                *self.ptr,
                url_or_filename.to_cstring().as_ptr(),
                ignore_ttl_setting as i32,
                None,
                std::ptr::null_mut(),
            )
        } != 0;
        check_result!(!result, "Download failed.");
        check_result!(self);
        Ok(result)
    }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_is_network_enabled() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        assert!(ctx.is_network_enabled()?);
        Ok(())
    }
    #[test]
    fn test_set_url_endpoint() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.set_url_endpoint("https://test.proj.org")?;
        let endpoint = ctx.get_url_endpoint()?;
        assert_eq!(endpoint, "https://test.proj.org");
        Ok(())
    }
    #[test]
    fn test_get_url_endpoint() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let endpoint = ctx.get_url_endpoint()?;
        assert_eq!(endpoint, "https://cdn.proj.org");
        Ok(())
    }
    #[test]
    fn test_get_user_writable_directory() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let dir = ctx.get_user_writable_directory(false)?;
        assert!(dir.to_str().unwrap().contains("proj"));
        Ok(())
    }
    #[test]
    fn test_grid_cache_set_enable() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.grid_cache_set_enable(false)?;
        Ok(())
    }
    #[test]
    fn test_grid_cache_set_max_size() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.grid_cache_set_max_size(1)?;
        Ok(())
    }
    #[test]
    fn test_grid_cache_set_ttl() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.grid_cache_set_ttl(1)?;
        Ok(())
    }
}
