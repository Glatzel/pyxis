use std::ffi::CString;
use std::path::PathBuf;

use miette::IntoDiagnostic;

use crate::check_result;

impl crate::Context {
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_network_callbacks>
    fn _set_network_callbacks(&self) { unimplemented!() }
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_enable_network>
    fn _set_enable_network(&self, enabled: bool) -> miette::Result<&Self> {
        let result =
            unsafe { proj_sys::proj_context_set_enable_network(self.ptr, enabled as i32) } != 0;
        if enabled ^ result {
            miette::bail!("Network interface is not available.")
        }
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_is_network_enabled>
    fn _is_network_enabled(&self) -> miette::Result<bool> {
        let result = unsafe { proj_sys::proj_context_is_network_enabled(self.ptr) } != 0;
        check_result!(self);
        Ok(result)
    }
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_url_endpoint>
    fn _set_url_endpoint(&self, url: &str) -> miette::Result<&Self> {
        unsafe {
            proj_sys::proj_context_set_url_endpoint(
                self.ptr,
                CString::new(url).into_diagnostic()?.as_ptr(),
            );
        };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_url_endpoint>
    fn _get_url_endpoint(&self) -> miette::Result<String> {
        let result = unsafe { proj_sys::proj_context_get_url_endpoint(self.ptr) };
        check_result!(self);
        Ok(crate::c_char_to_string(result).unwrap_or_default())
    }
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_context_get_user_writable_directory>
    fn _get_user_writable_directory(&self, create: bool) -> miette::Result<PathBuf> {
        let result =
            unsafe { proj_sys::proj_context_get_user_writable_directory(self.ptr, create as i32) };
        check_result!(self);
        Ok(PathBuf::from(
            crate::c_char_to_string(result).unwrap_or_default(),
        ))
    }
    /// # References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_cache_set_enable>
    fn _grid_cache_set_enable(&self, enabled: bool) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_enable(self.ptr, enabled as i32) };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// <>
    fn _grid_cache_set_filename(&self, fullname: &str) -> miette::Result<&Self> {
        unsafe {
            proj_sys::proj_grid_cache_set_filename(
                self.ptr,
                CString::new(fullname).into_diagnostic()?.as_ptr(),
            )
        };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// <>
    fn _grid_cache_set_max_size(&self, max_size_mbyte: u16) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_max_size(self.ptr, max_size_mbyte as i32) };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// <>
    fn _grid_cache_set_ttl(&self, ttl_seconds: u16) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_ttl(self.ptr, ttl_seconds as i32) };
        check_result!(self);
        Ok(self)
    }
    /// # References
    /// <>
    fn _grid_cache_clear(&self) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_clear(self.ptr) };
        Ok(self)
    }
    /// # References
    /// <>
    fn _is_download_needed(
        &self,
        url_or_filename: &str,
        ignore_ttl_setting: bool,
    ) -> miette::Result<bool> {
        let result = unsafe {
            proj_sys::proj_is_download_needed(
                self.ptr,
                CString::new(url_or_filename).into_diagnostic()?.as_ptr(),
                ignore_ttl_setting as i32,
            )
        } != 0;
        check_result!(self);
        Ok(result)
    }
    /// # References
    /// <>
    fn _download_file(
        &self,
        // url_or_filename: &str,
        // ignore_ttl_setting: bool,
        // progress_cbk: Option<unsafe extern "C" fn(arg1: Context, pct: f64, user_data: &mut T)>,
        // user_data: Option<T>,
    ) -> miette::Result<bool> {
        unimplemented!()
        // let result = unsafe {
        //     proj_sys::proj_download_file(
        //         self.ctx,
        //         CString::new(url_or_filename).into_diagnostic()?.as_ptr(),
        //         ignore_ttl_setting as i32,
        //         progress_cbk,
        //         user_data,
        //     )
        // } != 0;
        // if !result {
        //     miette::bail!("Download failed.")
        // }
        // check_result!(self);
        // Ok(result)
    }
}
