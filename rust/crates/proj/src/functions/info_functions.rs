use envoy::{CStrToString, ToCStr};

use crate::data_types::{GridInfo, Info, InitInfo, ProjInfo};

/// Get information about the current instance of the PROJ library.
///
/// References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_info>
pub fn info() -> Info {
    let src = unsafe { proj_sys::proj_info() };
    Info::new(
        src.major,
        src.minor,
        src.patch,
        src.release.to_string().unwrap_or_default(),
        src.version.to_string().unwrap_or_default(),
        src.searchpath.to_string().unwrap_or_default(),
    )
}
///# Info functions
impl crate::Proj<'_> {
    /// Get information about a specific grid.
    ///
    /// References
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
/// References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_grid_info>
pub fn grid_info(grid: &str) -> miette::Result<GridInfo> {
    let src = unsafe { proj_sys::proj_grid_info(grid.to_cstr()?) };
    if src.format.to_string().unwrap_or_default() == "missing" {
        miette::bail!("Invalid grid: {}", grid)
    }
    Ok(GridInfo::new(
        src.gridname.to_string().unwrap_or_default(),
        src.filename.to_string().unwrap_or_default(),
        src.format.to_string().unwrap_or_default(),
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
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_init_info>
pub fn init_info(initname: &str) -> miette::Result<InitInfo> {
    let src = unsafe { proj_sys::proj_init_info(initname.to_cstr()?) };
    let info = InitInfo::new(
        src.name.to_string().unwrap_or_default(),
        src.filename.to_string().unwrap_or_default(),
        src.version.to_string().unwrap_or_default(),
        src.origin.to_string().unwrap_or_default(),
        src.lastupdate.to_string().unwrap_or_default(),
    );
    if info.version() == "" {
        miette::bail!(format!("Invalid proj init file or name: {}", initname))
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

    use miette::IntoDiagnostic;

    use super::*;
    #[test]
    fn test_ctx_info() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
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
