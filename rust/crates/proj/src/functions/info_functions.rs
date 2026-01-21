use envoy::{PtrToString, ToCString};

use crate::data_types::{GridInfo, Info, InitInfo, ProjError, ProjErrorCode, ProjInfo};

/// Get information about the current instance of the PROJ library.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_info>
pub fn info() -> Result<Info, ProjError> {
    let src = unsafe { proj_sys::proj_info() };
    Ok(Info::new(
        src.major,
        src.minor,
        src.patch,
        src.release.to_string()?,
        src.version.to_string()?,
        src.searchpath.to_string()?,
    ))
}
///# Info functions
impl crate::Proj {
    /// Get information about a specific grid.
    ///
    /// References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_pj_info>
    pub fn info(&self) -> ProjInfo {
        let src = unsafe { proj_sys::proj_pj_info(self.ptr()) };
        ProjInfo::new(
            src.id.to_string().unwrap_or_default(),
            src.description.to_string().unwrap_or_default(),
            src.definition.to_string().unwrap_or_default(),
            src.has_inverse != 0,
            src.accuracy,
        )
    }
}

/// Get information about a specific grid.
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_info>
pub fn grid_info(grid: &str) -> Result<GridInfo, ProjError> {
    let src = unsafe { proj_sys::proj_grid_info(grid.to_cstring()?.as_ptr()) };
    if src.gridname.to_string().unwrap().as_str() == ""
        && src.filename.to_string().unwrap().as_str() == ""
        && src.format.to_string().unwrap_or_default() == "missing"
    {
        return Err(ProjError {
            code: ProjErrorCode::Other,
            message: format!("Invalid grid: {}", grid),
        });
    }
    Ok(GridInfo::new(
        src.gridname.to_string().unwrap(),
        src.filename.to_string().unwrap(),
        src.format.to_string().unwrap(),
        (src.lowerleft.lam, src.lowerleft.phi),
        (src.upperright.lam, src.upperright.phi),
        src.n_lon,
        src.n_lat,
        src.cs_lon,
        src.cs_lat,
    ))
}
/// Get information about a specific init file.
///
/// # Arguments
///
/// * `initname`: Init file in the PROJ searchpath
///
/// # References
///
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_init_info>
pub fn init_info(initname: &str) -> Result<InitInfo, ProjError> {
    let src = unsafe { proj_sys::proj_init_info(initname.to_cstring()?.as_ptr()) };
    let info = InitInfo::new(
        src.name.to_string().unwrap_or_default(),
        src.filename.to_string().unwrap_or_default(),
        src.version.to_string().unwrap_or_default(),
        src.origin.to_string().unwrap_or_default(),
        src.lastupdate.to_string().unwrap_or_default(),
    );
    if info.version() == "" {
        return Err(ProjError {
            code: ProjErrorCode::Other,
            message: format!("Invalid proj init file or name: {initname}"),
        });
    }
    Ok(InitInfo::new(
        src.name.to_string().unwrap_or_default(),
        src.filename.to_string().unwrap_or_default(),
        src.version.to_string().unwrap_or_default(),
        src.origin.to_string().unwrap_or_default(),
        src.lastupdate.to_string().unwrap_or_default(),
    ))
}
#[cfg(test)]
mod test {
    use std::path::PathBuf;

    use super::*;
    #[test]
    fn test_ctx_info() -> mischief::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        println!("{:?}", pj.info());
        assert_eq!(pj.info().description(), "WGS 84");
        assert!(!pj.info().has_inverse());
        Ok(())
    }
    #[test]
    fn test_pj_info() -> mischief::Result<()> {
        let info = info()?;
        println!("{info:?}");
        assert_eq!(info.major(), &(crate::version::PROJ_VERSION_MAJOR as i32));
        assert_eq!(info.minor(), &(crate::version::PROJ_VERSION_MINOR as i32));
        assert_eq!(info.patch(), &(crate::version::PROJ_VERSION_PATCH as i32));
        Ok(())
    }

    #[test]
    fn test_grid_info() -> mischief::Result<()> {
        //valid
        {
            let workspace_dir = PathBuf::from(std::env::var("CARGO_WORKSPACE_DIR")?);
            let info = grid_info(
                workspace_dir
                    .join("external/ntv2-file-routines/samples/mne.gsb")
                    .to_str()
                    .unwrap(),
            )?;
            println!("{info:?}");
            assert_eq!(info.format(), "ntv2");
        }
        // invalid
        {
            let info = grid_info("invalid.tiff");
            assert!(info.is_err());
        }
        Ok(())
    }
    #[test]
    fn test_init_info() -> mischief::Result<()> {
        //valid
        {
            let info = init_info("ITRF2000")?;
            println!("{info:?}");
            assert_eq!(info.name(), "ITRF2000");
        }
        //invalid
        {
            let info = init_info("invalid init");
            assert!(info.is_err());
        }
        Ok(())
    }
}
