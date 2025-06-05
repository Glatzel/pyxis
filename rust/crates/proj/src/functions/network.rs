use std::path::PathBuf;

use envoy::{CStrToString, ToCString};

use crate::check_result;
impl crate::Context {
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_network_callbacks>
    fn _set_network_callbacks(&self) { todo!() }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_enable_network>
    pub fn set_enable_network(&self, enabled: bool) -> miette::Result<&Self> {
        let result =
            unsafe { proj_sys::proj_context_set_enable_network(self.ptr, enabled as i32) } != 0;
        if enabled ^ result {
            miette::bail!("Network interface is not available.")
        }
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_is_network_enabled>
    pub fn is_network_enabled(&self) -> miette::Result<bool> {
        let result = unsafe { proj_sys::proj_context_is_network_enabled(self.ptr) } != 0;
        check_result!(self);
        Ok(result)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_url_endpoint>
    pub fn set_url_endpoint(&self, url: &str) -> miette::Result<&Self> {
        unsafe {
            proj_sys::proj_context_set_url_endpoint(self.ptr, url.to_cstring().as_ptr());
        };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_url_endpoint>
    pub fn get_url_endpoint(&self) -> miette::Result<String> {
        let result = unsafe { proj_sys::proj_context_get_url_endpoint(self.ptr) };
        check_result!(self);
        Ok(result.to_string().unwrap_or_default())
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_user_writable_directory>
    pub fn get_user_writable_directory(&self, create: bool) -> miette::Result<PathBuf> {
        let result =
            unsafe { proj_sys::proj_context_get_user_writable_directory(self.ptr, create as i32) };
        check_result!(self);
        Ok(PathBuf::from(result.to_string().unwrap_or_default()))
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_enable>
    pub fn grid_cache_set_enable(&self, enabled: bool) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_enable(self.ptr, enabled as i32) };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_filename>
    pub fn grid_cache_set_filename(&self, fullname: &str) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_filename(self.ptr, fullname.to_cstring().as_ptr()) };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_max_size>
    pub fn grid_cache_set_max_size(&self, max_size_mbyte: u16) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_max_size(self.ptr, max_size_mbyte as i32) };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_ttl>
    pub fn grid_cache_set_ttl(&self, ttl_seconds: u16) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_ttl(self.ptr, ttl_seconds as i32) };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_clear>
    pub fn grid_cache_clear(&self) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_clear(self.ptr) };
        Ok(self)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_is_download_needed>
    pub fn is_download_needed(
        &self,
        url_or_filename: &str,
        ignore_ttl_setting: bool,
    ) -> miette::Result<bool> {
        let result = unsafe {
            proj_sys::proj_is_download_needed(
                self.ptr,
                url_or_filename.to_cstring().as_ptr(),
                ignore_ttl_setting as i32,
            )
        } != 0;
        check_result!(self);
        Ok(result)
    }
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_download_file>
    pub fn download_file(
        &self,
        url_or_filename: &str,
        ignore_ttl_setting: bool,
    ) -> miette::Result<bool> {
        let result = unsafe {
            proj_sys::proj_download_file(
                self.ptr,
                url_or_filename.to_cstring().as_ptr(),
                ignore_ttl_setting as i32,
                None,
                std::ptr::null_mut(),
            )
        } != 0;
        if !result {
            miette::bail!("Download failed.")
        }
        check_result!(self);
        Ok(result)
    }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_is_network_enabled() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        assert!(ctx.is_network_enabled()?);
        Ok(())
    }
    #[test]
    fn test_set_url_endpoint() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.set_url_endpoint("https://cdn.proj.org")?;
        Ok(())
    }
    #[test]
    fn test_get_url_endpoint() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let endpoint = ctx.get_url_endpoint()?;
        assert_eq!(endpoint, "https://cdn.proj.org");
        Ok(())
    }
    #[test]
    fn test_get_user_writable_directory() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let dir = ctx.get_user_writable_directory(false)?;
        assert!(dir.to_str().unwrap().contains("proj"));
        Ok(())
    }
    #[test]
    fn test_grid_cache_set_enable() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.grid_cache_set_enable(false)?;
        Ok(())
    }
    #[test]
    fn test_grid_cache_set_max_size() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.grid_cache_set_max_size(1)?;
        Ok(())
    }
    #[test]
    fn test_grid_cache_set_ttl() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.grid_cache_set_ttl(1)?;
        Ok(())
    }
}
