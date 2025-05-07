use miette::IntoDiagnostic;

use crate::data_types::{PjGridInfo, PjInfo, PjInitInfo, PjProjInfo};

/// Get information about the current instance of the PROJ library.
///
/// References
/// <https://proj.org/en/stable/development/reference/functions.html#c.proj_info>
pub fn info() -> PjInfo {
    let src = unsafe { proj_sys::proj_info() };
    PjInfo::new(
        src.major,
        src.minor,
        src.patch,
        crate::c_char_to_string(src.release),
        crate::c_char_to_string(src.version),
        crate::c_char_to_string(src.searchpath),
    )
}
///# Info functions
impl crate::Proj {
    /// Get information about a specific grid.
    ///
    /// References
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_pj_info>
    pub fn info(&self) -> PjProjInfo {
        let src = unsafe { proj_sys::proj_pj_info(self.pj) };
        PjProjInfo::new(
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
pub fn grid_info(grid: &str) -> miette::Result<PjGridInfo> {
    let gridname_cstr = std::ffi::CString::new(grid).into_diagnostic()?;
    let src = unsafe { proj_sys::proj_grid_info(gridname_cstr.as_ptr()) };
    if crate::c_char_to_string(src.format.as_ptr()) == "missing" {
        miette::bail!("Invalid grid: {}", grid)
    }
    Ok(PjGridInfo::new(
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
pub fn init_info(initname: &str) -> miette::Result<PjInitInfo> {
    let initname_cstr = std::ffi::CString::new(initname).into_diagnostic()?;
    let src = unsafe { proj_sys::proj_init_info(initname_cstr.as_ptr()) };
    let info = PjInitInfo::new(
        crate::c_char_to_string(src.name.as_ptr()),
        crate::c_char_to_string(src.filename.as_ptr()),
        crate::c_char_to_string(src.version.as_ptr()),
        crate::c_char_to_string(src.origin.as_ptr()),
        crate::c_char_to_string(src.lastupdate.as_ptr()),
    );
    if info.version() == "" {
        miette::bail!(format!("Invalid proj init file or name: {}", initname))
    }
    Ok(PjInitInfo::new(
        crate::c_char_to_string(src.name.as_ptr()),
        crate::c_char_to_string(src.filename.as_ptr()),
        crate::c_char_to_string(src.version.as_ptr()),
        crate::c_char_to_string(src.origin.as_ptr()),
        crate::c_char_to_string(src.lastupdate.as_ptr()),
    ))
}
#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use super::*;
    #[test]
    fn test_ctx_info() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        let pj = ctx.create("EPSG:4326")?;
        println!("{:?}", pj.info());
        Ok(())
    }
    #[test]
    fn test_pj_info() {
        let info = info();
        println!("{:?}", info);
    }

    #[test]
    fn test_grid_info_gsb() -> miette::Result<()> {
        let workspace_dir = PathBuf::from(std::env::var("CARGO_WORKSPACE_DIR").into_diagnostic()?);
        let info = grid_info(
            workspace_dir
                .join("external/ntv2-file-routines/samples/mne.gsb")
                .to_str()
                .unwrap(),
        )?;
        println!("{:?}", info);
        assert_eq!(info.format(), "ntv2");
        Ok(())
    }

    #[test]
    fn test_grid_info_invalid_grid() -> miette::Result<()> {
        let info = grid_info("Cargo.toml");
        assert!(info.is_err());
        Ok(())
    }

    #[test]
    fn test_grid_info_not_exists() -> miette::Result<()> {
        let info = grid_info("invalid.tiff");
        assert!(info.is_err());
        Ok(())
    }

    #[test]
    fn test_init_info() -> miette::Result<()> {
        let info = init_info("ITRF2000")?;
        println!("{:?}", info);
        Ok(())
    }

    #[test]
    fn test_init_info_fail() -> miette::Result<()> {
        let info = init_info("invalid init");
        assert!(info.is_err());
        Ok(())
    }
}
