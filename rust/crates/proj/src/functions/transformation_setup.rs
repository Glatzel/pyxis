use std::ffi::CString;

use miette::IntoDiagnostic;

use crate::check_result;
/// # Transformation setup
impl crate::PjContext {
    /// See [`Self::create_proj`], [`crate::PjParams::Definition`]
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create>
    pub fn create(&self, definition: &str) -> miette::Result<crate::Pj> {
        let definition = CString::new(definition).into_diagnostic()?;
        let pj = crate::Pj {
            ptr: unsafe { proj_sys::proj_create(self.ptr, definition.as_ptr()) },
            ctx: self,
        };
        check_result!(self);
        Ok(pj)
    }
    /// See [`Self::create_proj`], [`crate::PjParams::Argv`]
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv>
    pub fn create_argv(&self, argv: &[&str]) -> miette::Result<crate::Pj> {
        let len = argv.len();
        let mut ptrs: Vec<*mut i8> = Vec::with_capacity(len);
        for s in argv {
            ptrs.push(CString::new(*s).into_diagnostic()?.into_raw());
        }
        let pj = crate::Pj {
            ptr: unsafe { proj_sys::proj_create_argv(self.ptr, len as i32, ptrs.as_mut_ptr()) },
            ctx: self,
        };
        check_result!(self);
        Ok(pj)
    }
    /// See [`Self::create_proj`], [`crate::PjParams::CrsToCrs`]
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs>
    pub fn create_crs_to_crs(
        &self,
        source_crs: &str,
        target_crs: &str,
        area: &crate::PjArea,
    ) -> miette::Result<crate::Pj> {
        let source_crs = CString::new(source_crs).into_diagnostic()?;
        let target_crs = CString::new(target_crs).into_diagnostic()?;
        let pj = crate::Pj {
            ptr: unsafe {
                proj_sys::proj_create_crs_to_crs(
                    self.ptr,
                    source_crs.as_ptr(),
                    target_crs.as_ptr(),
                    area.ptr,
                )
            },
            ctx: self,
        };
        check_result!(self);
        Ok(pj)
    }
    /// See [`Self::create_proj`], [`crate::PjParams::CrsToCrsFromPj`]
    ///
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs_from_pj>
    pub fn create_crs_to_crs_from_pj(
        &self,
        source_crs: crate::Pj,
        target_crs: crate::Pj,
        area: &crate::PjArea,
        authority: Option<&str>,
        accuracy: Option<f64>,
        allow_ballpark: Option<bool>,
        only_best: Option<bool>,
        force_over: Option<bool>,
    ) -> miette::Result<crate::Pj> {
        let mut options = crate::PjOptions::new(5);
        options
            .push_optional_pass(authority, "AUTHORITY")
            .push_optional_pass(accuracy, "ACCURACY")
            .push_optional_pass(allow_ballpark, "ALLOW_BALLPARK")
            .push_optional_pass(only_best, "ONLY_BEST")
            .push_optional_pass(force_over, "FORCE_OVER");

        let pj = crate::Pj {
            ptr: unsafe {
                proj_sys::proj_create_crs_to_crs_from_pj(
                    self.ptr,
                    source_crs.ptr,
                    target_crs.ptr,
                    area.ptr,
                    options.as_ptr(),
                )
            },
            ctx: self,
        };
        check_result!(self);
        Ok(pj)
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_normalize_for_visualization>
    pub fn normalize_for_visualization(&self, obj: &crate::Pj) -> miette::Result<crate::Pj> {
        Ok(crate::Pj {
            ptr: unsafe { proj_sys::proj_normalize_for_visualization(self.ptr, obj.ptr) },
            ctx: self,
        })
    }
}

impl Drop for crate::Pj<'_> {
    fn drop(&mut self) { unsafe { proj_sys::proj_destroy(self.ptr) }; }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_create() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        assert!(!pj.ptr.is_null());
        Ok(())
    }

    #[test]
    fn test_create_argv() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_argv(&["proj=utm", "zone=32", "ellps=GRS80"])?;
        assert!(!pj.ptr.is_null());
        Ok(())
    }

    #[test]
    fn test_create_crs_to_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        assert!(!pj.ptr.is_null());
        Ok(())
    }

    #[test]
    fn test_create_crs_to_crs_from_pj() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj1 = ctx.create("EPSG:4326")?;
        let pj2 = ctx.create("EPSG:4978")?;

        let pj3 = ctx.create_crs_to_crs_from_pj(
            pj1,
            pj2,
            &crate::PjArea::default(),
            Some("any"),
            Some(0.001),
            Some(true),
            Some(true),
            Some(true),
        )?;
        assert!(!pj3.ptr.is_null());
        Ok(())
    }
}
