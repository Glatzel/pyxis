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
                    area.area,
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
        let mut options: Vec<*const i8> = Vec::with_capacity(5);
        if let Some(authority) = authority {
            options.push(
                CString::new(format!("AUTHORITY={}", authority))
                    .into_diagnostic()?
                    .as_ptr(),
            );
        }
        if let Some(accuracy) = accuracy {
            options.push(
                CString::new(format!("ACCURACY={}", accuracy))
                    .into_diagnostic()?
                    .as_ptr(),
            );
        }
        if let Some(allow_ballpark) = allow_ballpark {
            options.push(
                CString::new(format!(
                    "ALLOW_BALLPARK={}",
                    if allow_ballpark { "YES" } else { "NO" }
                ))
                .into_diagnostic()?
                .as_ptr(),
            );
        }
        if let Some(only_best) = only_best {
            options.push(
                CString::new(format!(
                    "ONLY_BEST={}",
                    if only_best { "YES" } else { "NO" }
                ))
                .into_diagnostic()?
                .as_ptr(),
            );
        }
        if let Some(force_over) = force_over {
            options.push(
                CString::new(format!(
                    "FORCE_OVER={}",
                    if force_over { "YES" } else { "NO" }
                ))
                .into_diagnostic()?
                .as_ptr(),
            );
        }
        let pj = crate::Pj {
            ptr: unsafe {
                proj_sys::proj_create_crs_to_crs_from_pj(
                    self.ptr,
                    source_crs.ptr,
                    target_crs.ptr,
                    area.area,
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
        ctx.create("EPSG:4326")?;
        Ok(())
    }

    #[test]
    fn test_create_argv() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.create_argv(&["proj=utm", "zone=32", "ellps=GRS80"])?;
        Ok(())
    }

    #[test]
    fn test_create_crs_to_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.create_crs_to_crs("EPSG:4326", "EPSG:4978", &crate::PjArea::default())?;
        Ok(())
    }

    #[test]
    fn test_create_crs_to_crs_from_pj() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj1 = ctx.create("EPSG:4326")?;
        let pj2 = ctx.create("EPSG:4978")?;
        ctx.create_crs_to_crs_from_pj(
            pj1,
            pj2,
            &crate::PjArea::default(),
            Some("any"),
            Some(0.001),
            Some(true),
            Some(true),
            Some(true),
        )?;
        Ok(())
    }
}
