///Major version number, e.g. 8 for PROJ 8.0.1
pub const PROJ_VERSION_MAJOR: u32 = proj_sys::PROJ_VERSION_MAJOR;
///Minor version number, e.g. 0 for PROJ 8.0.1
pub const PROJ_VERSION_MINOR: u32 = proj_sys::PROJ_VERSION_MINOR;
///Patch version number, e.g. 1 for PROJ 8.0.1
pub const PROJ_VERSION_PATCH: u32 = proj_sys::PROJ_VERSION_PATCH;
pub fn compute_version(maj: u32, min: u32, patch: u32) -> u32 {
    (maj) * 10000 + (min) * 100 + (patch)
}
pub const PROJ_VERSION_NUMBER: u32 =
    PROJ_VERSION_MAJOR * 10000 + PROJ_VERSION_MINOR * 100 + PROJ_VERSION_PATCH;
pub fn at_least_version(maj: u32, min: u32, patch: u32) -> bool {
    PROJ_VERSION_NUMBER >= compute_version(maj, min, patch)
}
#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_proj_version_number() {
        assert_eq!(PROJ_VERSION_NUMBER, 90600);
    }
}
