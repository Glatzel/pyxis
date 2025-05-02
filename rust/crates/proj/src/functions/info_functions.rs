use miette::IntoDiagnostic;

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
pub fn grid_info(gridname: &str) -> miette::Result<crate::PjGridInfo> {
    let gridname = std::ffi::CString::new(gridname).into_diagnostic()?;
    let src = unsafe { proj_sys::proj_grid_info(gridname.as_ptr()) };
    Ok(crate::PjGridInfo::new(
        crate::c_char_to_string(src.gridname.as_ptr()),
        crate::c_char_to_string(src.filename.as_ptr()),
        crate::c_char_to_string(src.format.as_ptr()),
        src.n_lon,
        src.n_lat,
        src.cs_lon,
        src.cs_lat,
    ))
}
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
    // region:Info functions
    #[test]
    fn test_info() {
        let info = info();
        println!("{:?}", info);
    }
}
