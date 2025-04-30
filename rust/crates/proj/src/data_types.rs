// region:Transformation objects
pub struct Proj {
    pj: *mut proj_sys::PJ,
}
impl Drop for Proj {
    fn drop(&mut self) {
        unsafe { proj_sys::proj_destroy(self.pj) };
    }
}
// region:Coordinate transformation
///https://proj.org/en/stable/development/reference/functions.html#coordinate-transformation
impl Proj {
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans
    pub fn trans(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_get_last_used_operation
    pub fn trans_get_last_used_operation(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_generic
    pub fn proj_trans_generic(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_array
    pub fn proj_trans_array(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds
    pub fn proj_trans_bounds(&self) {
        unimplemented!()
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_trans_bounds_3D
    pub fn proj_trans_bounds_3d(&self) {
        unimplemented!()
    }
}
struct _ProjDirection {}

pub struct Context {
    ctx: *mut proj_sys::PJ_CONTEXT,
}
// region:Threading contexts
impl Context {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_create
    pub fn new() -> Self {
        Self {
            ctx: unsafe { proj_sys::proj_context_create() },
        }
    }
}
impl Clone for Context {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_clone
    fn clone(&self) -> Self {
        Self {
            ctx: unsafe { proj_sys::proj_context_clone(self.ctx) },
        }
    }
}
impl Drop for Context {
    /// #References
    /// https://proj.org/en/stable/development/reference/functions.html#c.proj_context_destroy
    fn drop(&mut self) {
        unsafe { proj_sys::proj_context_destroy(self.ctx) };
    }
}
// region:Transformation setup
///https://proj.org/en/stable/development/reference/functions.html#transformation-setup
impl Context {
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_create
    pub fn proj_create(&self, definition: &str) -> Proj {
        Proj {
            pj: unsafe { proj_sys::proj_create(self.ctx, definition.as_ptr() as *const i8) },
        }
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv
    pub fn proj_create_argv(&self, definition: &[&str]) -> Proj {
        let len = definition.len();
        let mut ptrs: Vec<*mut i8> = Vec::with_capacity(len);
        for s in definition {
            let c_str: std::ffi::CString = std::ffi::CString::new(*s).expect("CString::new failed");
            let ptr = c_str.as_ptr() as *mut i8; // Convert to *mut i8
            ptrs.push(ptr);
        }
        Proj {
            pj: unsafe { proj_sys::proj_create_argv(self.ctx, len as i32, ptrs.as_mut_ptr()) },
        }
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs
    pub fn proj_create_crs_to_crs(&self, source_crs: &str, target_crs: &str, area: Area) -> Proj {
        Proj {
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
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs_from_pj
    pub fn proj_create_crs_to_crs_from_pj(
        &self,
        source_crs: Proj,
        target_crs: Proj,
        area: Area,
        authority: Option<&str>,
        accuracy: Option<f64>,
        allow_ballpark: Option<bool>,
        only_best: Option<bool>,
        force_over: Option<bool>,
    ) -> Proj {
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
        Proj {
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
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_normalize_for_visualization
    fn _normalize_for_visualization() {
        unimplemented!()
    }
}

pub struct Area {
    area: *mut proj_sys::PJ_AREA,
}
impl Area {
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_area_create
    pub fn new() -> Self {
        Self {
            area: unsafe { proj_sys::proj_area_create() },
        }
    }
    ///https://proj.org/en/stable/development/reference/functions.html#c.proj_area_set_bbox
    pub fn set_bbox(
        &self,
        west_lon_degree: f64,
        south_lat_degree: f64,
        east_lon_degree: f64,
        north_lat_degree: f64,
    ) -> &Self {
        unsafe {
            proj_sys::proj_area_set_bbox(
                self.area,
                west_lon_degree,
                south_lat_degree,
                east_lon_degree,
                north_lat_degree,
            )
        };
        self
    }
}
