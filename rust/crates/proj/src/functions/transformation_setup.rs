//!The objects returned by the functions defined in this section have minimal
//! interaction with the functions of the C API for ISO-19111 functionality, and
//! vice versa. See its introduction paragraph for more details.
//!
//! # References
//!
//!* <https://proj.org/en/stable/development/reference/functions.html#c-api-for-iso-19111-functionality>

use std::ffi::CString;

use miette::IntoDiagnostic;

use crate::{Proj, check_result};
/// # Transformation setup
impl crate::Context {
    /// # References
    ///* <https://proj.org/en/stable/development/reference/functions.html#c.proj_create>
    pub fn create(&self, definition: &str) -> miette::Result<crate::Proj> {
        let definition = CString::new(definition).into_diagnostic()?;
        let ptr = unsafe { proj_sys::proj_create(self.ptr, definition.as_ptr()) };
        check_result!(self);
        Proj::from_raw(self, ptr)
    }

    /// # References
    ///* <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv>
    pub fn create_argv(&self, argv: &[&str]) -> miette::Result<crate::Proj> {
        let len = argv.len();
        let mut argv_ptrs: Vec<*mut i8> = Vec::with_capacity(len);
        for s in argv {
            argv_ptrs.push(CString::new(*s).into_diagnostic()?.into_raw());
        }
        let ptr =
            unsafe { proj_sys::proj_create_argv(self.ptr, len as i32, argv_ptrs.as_mut_ptr()) };
        check_result!(self);
        Proj::from_raw(self, ptr)
    }

    /// # References
    ///* <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs>
    pub fn create_crs_to_crs(
        &self,
        source_crs: &str,
        target_crs: &str,
        area: &crate::Area,
    ) -> miette::Result<crate::Proj> {
        let source_crs = CString::new(source_crs).into_diagnostic()?;
        let target_crs = CString::new(target_crs).into_diagnostic()?;
        let ptr = unsafe {
            proj_sys::proj_create_crs_to_crs(
                self.ptr,
                source_crs.as_ptr(),
                target_crs.as_ptr(),
                area.ptr,
            )
        };
        check_result!(self);
        Proj::from_raw(self, ptr)
    }

    /// # References
    ///* <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs_from_pj>
    pub fn create_crs_to_crs_from_pj(
        &self,
        source_crs: crate::Proj,
        target_crs: crate::Proj,
        area: &crate::Area,
        authority: Option<&str>,
        accuracy: Option<f64>,
        allow_ballpark: Option<bool>,
        only_best: Option<bool>,
        force_over: Option<bool>,
    ) -> miette::Result<crate::Proj> {
        let mut options = crate::ProjOptions::new(5);
        options
            .push_optional_pass(authority, "AUTHORITY")
            .push_optional_pass(accuracy, "ACCURACY")
            .push_optional_pass(allow_ballpark, "ALLOW_BALLPARK")
            .push_optional_pass(only_best, "ONLY_BEST")
            .push_optional_pass(force_over, "FORCE_OVER");
        let ptrs = options.vec_ptr();
        let ptr = unsafe {
            proj_sys::proj_create_crs_to_crs_from_pj(
                self.ptr,
                source_crs.ptr,
                target_crs.ptr,
                area.ptr,
                ptrs.as_ptr(),
            )
        };
        check_result!(self);
        Proj::from_raw(self, ptr)
    }
    /// # References
    ///* <https://proj.org/en/stable/development/reference/functions.html#c.proj_normalize_for_visualization>
    pub fn normalize_for_visualization(&self, obj: &crate::Proj) -> miette::Result<crate::Proj> {
        let ptr = unsafe { proj_sys::proj_normalize_for_visualization(self.ptr, obj.ptr) };
        Proj::from_raw(self, ptr)
    }
}

impl Drop for crate::Proj<'_> {
    fn drop(&mut self) { unsafe { proj_sys::proj_destroy(self.ptr) }; }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_create() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let _ = pj.clone();
        Ok(())
    }

    #[test]
    fn test_create_argv() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let _ = ctx.create_argv(&["proj=utm", "zone=32", "ellps=GRS80"])?;
        Ok(())
    }

    #[test]
    fn test_create_crs_to_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let _ = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::Area::default())?;
        Ok(())
    }

    #[test]
    fn test_create_crs_to_crs_from_pj() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj1 = ctx.create("EPSG:4326")?;
        let pj2 = ctx.create("EPSG:4978")?;
        let _ = ctx.create_crs_to_crs_from_pj(
            pj1,
            pj2,
            &crate::Area::default(),
            Some("any"),
            Some(0.001),
            Some(true),
            Some(true),
            Some(true),
        )?;
        Ok(())
    }
}
