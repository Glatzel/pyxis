use std::ffi::CString;
use std::path::PathBuf;

use miette::IntoDiagnostic;

use crate::check_result;

impl crate::PjContext {
    pub fn set_network_callbacks(&self,) { unimplemented!() }
    pub fn set_enable_network(&self, enabled: bool) -> miette::Result<&Self> {
        let result =
            unsafe { proj_sys::proj_context_set_enable_network(self.ctx, enabled as i32) } != 0;
        if enabled ^ result {
            miette::bail!("Network interface is not available.")
        }
        check_result!(self);
        Ok(self)
    }
    pub fn is_network_enabled(&self) -> miette::Result<bool> {
        let result = unsafe { proj_sys::proj_context_is_network_enabled(self.ctx) } != 0;
        check_result!(self);
        Ok(result)
    }
    pub fn set_url_endpoint(&self, url: &str) -> miette::Result<&Self> {
        unsafe {
            proj_sys::proj_context_set_url_endpoint(
                self.ctx,
                CString::new(url).into_diagnostic()?.as_ptr(),
            );
        };
        check_result!(self);
        Ok(self)
    }
    pub fn get_url_endpoint(&self) -> miette::Result<String> {
        let result = unsafe { proj_sys::proj_context_get_url_endpoint(self.ctx) };
        check_result!(self);
        Ok(crate::c_char_to_string(result))
    }
    pub fn get_user_writable_directory(&self, create: bool) -> miette::Result<PathBuf> {
        let result =
            unsafe { proj_sys::proj_context_get_user_writable_directory(self.ctx, create as i32) };
        check_result!(self);
        Ok(PathBuf::from(crate::c_char_to_string(result)))
    }
    pub fn grid_cache_set_enable(&self, enabled: bool) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_enable(self.ctx, enabled as i32) };
        check_result!(self);
        Ok(self)
    }
    pub fn grid_cache_set_filename(&self, fullname: &str) -> miette::Result<&Self> {
        unsafe {
            proj_sys::proj_grid_cache_set_filename(
                self.ctx,
                CString::new(fullname).into_diagnostic()?.as_ptr(),
            )
        };
        check_result!(self);
        Ok(self)
    }
    pub fn grid_cache_set_max_size(&self, max_size_mbyte: u16) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_max_size(self.ctx, max_size_mbyte as i32) };
        check_result!(self);
        Ok(self)
    }
    pub fn grid_cache_set_ttl(&self, ttl_seconds: u16) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_set_ttl(self.ctx, ttl_seconds as i32) };
        check_result!(self);
        Ok(self)
    }
    pub fn grid_cache_clear(&self) -> miette::Result<&Self> {
        unsafe { proj_sys::proj_grid_cache_clear(self.ctx) };
        Ok(self)
    }
    pub fn is_download_needed(
        &self,
        url_or_filename: &str,
        ignore_ttl_setting: bool,
    ) -> miette::Result<bool> {
        let result = unsafe {
            proj_sys::proj_is_download_needed(
                self.ctx,
                CString::new(url_or_filename).into_diagnostic()?.as_ptr(),
                ignore_ttl_setting as i32,
            )
        } != 0;
        check_result!(self);
        Ok(result)
    }
    pub fn download_file<T>(
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
