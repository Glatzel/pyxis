//! Proj version

///Major version number, e.g. 8 for PROJ 8.0.1
///
/// # Reference
///
/// * <https://proj.org/en/stable/development/reference/macros.html#c.PROJ_VERSION_MAJOR>
pub const PROJ_VERSION_MAJOR: u32 = proj_sys::PROJ_VERSION_MAJOR;

///Minor version number, e.g. 0 for PROJ 8.0.1
///
/// # Reference
///
/// * <https://proj.org/en/stable/development/reference/macros.html#c.PROJ_VERSION_MINOR>
pub const PROJ_VERSION_MINOR: u32 = proj_sys::PROJ_VERSION_MINOR;

///Patch version number, e.g. 1 for PROJ 8.0.1
///
/// # Reference
///
/// * <https://proj.org/en/stable/development/reference/macros.html#c.PROJ_VERSION_PATCH>
pub const PROJ_VERSION_PATCH: u32 = proj_sys::PROJ_VERSION_PATCH;

/// Compute the version number from the major, minor and patch numbers.
///
///# Reference
///
/// * <https://proj.org/en/stable/development/reference/macros.html#c.PROJ_COMPUTE_VERSION>
pub fn compute_version(maj: u32, min: u32, patch: u32) -> u32 {
    (maj) * 10000 + (min) * 100 + (patch)
}

///Total version number, equal to `PROJ_COMPUTE_VERSION(PROJ_VERSION_MAJOR,
/// PROJ_VERSION_MINOR,PROJ_VERSION_PATCH)`
///
///# Reference
///
/// * <https://proj.org/en/stable/development/reference/macros.html#c.PROJ_VERSION_NUMBER>
pub const PROJ_VERSION_NUMBER: u32 =
    PROJ_VERSION_MAJOR * 10000 + PROJ_VERSION_MINOR * 100 + PROJ_VERSION_PATCH;

///Macro that returns true if the current PROJ version is at least the version
/// specified by (maj,min,patch)
///
///Equivalent to `PROJ_VERSION_NUMBER >= PROJ_COMPUTE_VERSION(maj,min,patch)`
///
///# Reference
///
/// * <https://proj.org/en/stable/development/reference/macros.html#c.PROJ_AT_LEAST_VERSION>
pub fn at_least_version(maj: u32, min: u32, patch: u32) -> bool {
    PROJ_VERSION_NUMBER >= compute_version(maj, min, patch)
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_proj_version_number() {
        at_least_version(9, 7, 0);
        assert_eq!(PROJ_VERSION_NUMBER, 90602)
    }
}
