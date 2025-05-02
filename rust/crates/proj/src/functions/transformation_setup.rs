/// #References
///<https://proj.org/en/stable/development/reference/functions.html#transformation-setup>
impl crate::PjContext {
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create>
    pub fn proj_create(&self, definition: &str) -> crate::Pj {
        crate::Pj {
            pj: unsafe { proj_sys::proj_create(self.ctx, definition.as_ptr() as *const i8) },
        }
    }
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv>
    pub fn proj_create_argv(&self, definition: &[&str]) -> miette::Result<crate::Pj> {
        let len = definition.len();
        let mut ptrs: Vec<*mut i8> = Vec::with_capacity(len);
        for s in definition {
            ptrs.push(crate::string_to_c_char(s)?.cast_mut() as *mut i8);
        }
        Ok(crate::Pj {
            pj: unsafe { proj_sys::proj_create_argv(self.ctx, len as i32, ptrs.as_mut_ptr()) },
        })
    }
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs>
    pub fn proj_create_crs_to_crs(
        &self,
        source_crs: &str,
        target_crs: &str,
        area: crate::PjArea,
    ) -> crate::Pj {
        crate::Pj {
            pj: unsafe {
                proj_sys::proj_create_crs_to_crs(
                    self.ctx,
                    source_crs.as_ptr() as *const i8,
                    target_crs.as_ptr() as *const i8,
                    area.area,
                )
            },
        }
    }
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs_from_pj>
    pub fn proj_create_crs_to_crs_from_pj(
        &self,
        source_crs: crate::Pj,
        target_crs: crate::Pj,
        area: crate::PjArea,
        authority: Option<&str>,
        accuracy: Option<f64>,
        allow_ballpark: Option<bool>,
        only_best: Option<bool>,
        force_over: Option<bool>,
    ) -> crate::Pj {
        let mut options: Vec<*const i8> = Vec::with_capacity(5);
        if let Some(authority) = authority {
            options.push(format!("AUTHORITY={}", authority).as_ptr() as *mut i8);
        }
        if let Some(accuracy) = accuracy {
            options.push(format!("ACCURACY={}", accuracy).as_ptr() as *mut i8);
        }
        if let Some(allow_ballpark) = allow_ballpark {
            options.push(
                format!(
                    "ALLOW_BALLPARK={}",
                    if allow_ballpark { "YES" } else { "NO" }
                )
                .as_ptr() as *mut i8,
            );
        }
        if let Some(only_best) = only_best {
            options.push(
                format!("ONLY_BEST={}", if only_best { "YES" } else { "NO" }).as_ptr() as *mut i8,
            );
        }
        if let Some(force_over) = force_over {
            options.push(
                format!("FORCE_OVER={}", if force_over { "YES" } else { "NO" }).as_ptr() as *mut i8,
            );
        }
        crate::Pj {
            pj: unsafe {
                proj_sys::proj_create_crs_to_crs_from_pj(
                    self.ctx,
                    source_crs.pj,
                    target_crs.pj,
                    area.area,
                    options.as_ptr(),
                )
            },
        }
    }
    /// #References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_normalize_for_visualization>
    fn _normalize_for_visualization() {
        unimplemented!()
    }
}

impl Drop for crate::Pj {
    fn drop(&mut self) {
        unsafe { proj_sys::proj_destroy(self.pj) };
    }
}
