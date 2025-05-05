use miette::IntoDiagnostic;
/// Get information about the current instance of the PROJ library.
///
/// References
/// <https://proj.org/en/stable/development/reference/functions.html#c.proj_info>
pub fn info() -> crate::PjInfo {
    let src = unsafe { proj_sys::proj_info() };
    crate::PjInfo::new(
        src.major,
        src.minor,
        src.patch,
        crate::c_char_to_string(src.release),
        crate::c_char_to_string(src.version),
        crate::c_char_to_string(src.searchpath),
    )
}
///Info functions
impl crate::Pj {
    /// Get information about a specific grid.
    ///
    /// References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_pj_info>
    pub fn info(&self) -> crate::PjProjInfo {
        let src = unsafe { proj_sys::proj_pj_info(self.pj) };
        crate::PjProjInfo::new(
            crate::c_char_to_string(src.id),
            crate::c_char_to_string(src.description),
            crate::c_char_to_string(src.definition),
            src.has_inverse != 0,
            src.accuracy,
        )
    }
}

/// Get information about a specific grid.
///
/// References
/// <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_info>
pub fn grid_info(gridname: &str) -> miette::Result<crate::PjGridInfo> {
    let gridname = std::ffi::CString::new(gridname).into_diagnostic()?;
    let src = unsafe { proj_sys::proj_grid_info(gridname.as_ptr()) };
    Ok(crate::PjGridInfo::new(
        crate::c_char_to_string(src.gridname.as_ptr()),
        crate::c_char_to_string(src.filename.as_ptr()),
        crate::c_char_to_string(src.format.as_ptr()),
        src.lowerleft,
        src.upperright,
        src.n_lon,
        src.n_lat,
        src.cs_lon,
        src.cs_lat,
    ))
}
/// Get information about a specific init file.
///
/// References
/// <https://proj.org/en/stable/development/reference/functions.html#c.proj_init_info>
pub fn init_info(initname: &str) -> miette::Result<crate::PjInitInfo> {
    let initname = std::ffi::CString::new(initname).into_diagnostic()?;
    let src = unsafe { proj_sys::proj_init_info(initname.as_ptr()) };
    Ok(crate::PjInitInfo::new(
        crate::c_char_to_string(src.name.as_ptr()),
        crate::c_char_to_string(src.filename.as_ptr()),
        crate::c_char_to_string(src.version.as_ptr()),
        crate::c_char_to_string(src.origin.as_ptr()),
        crate::c_char_to_string(src.lastupdate.as_ptr()),
    ))
}
#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_ctx_info() -> miette::Result<()> {
        let ctx = crate::PjContext::default();
        let pj = ctx.create("EPSG:4326")?;
        println!("{:?}", pj.info());
        Ok(())
    }
    #[test]
    fn test_pj_info() {
        let info = info();
        println!("{:?}", info);
    }
}
