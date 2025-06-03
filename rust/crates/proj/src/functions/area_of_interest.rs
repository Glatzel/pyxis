impl Default for crate::Area {
    /// # See Also
    ///
    /// * [Self::new]
    fn default() -> Self { Self::new() }
}
///# Area of interest
impl crate::Area {
    ///Create an area of use.
    ///
    /// Such an area of use is to be passed to
    /// [`crate::Context::create_crs_to_crs`] to specify the area of use for the
    /// choice of relevant coordinate operations.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_area_create>
    pub fn new() -> Self {
        Self {
            ptr: unsafe { proj_sys::proj_area_create() },
        }
    }
    ///Set the bounding box of the area of use
    ///
    /// Such an area of use is to be passed to
    /// [`crate::Context::create_crs_to_crs`] to specify the area of use for the
    /// choice of relevant coordinate operations.
    ///
    /// In the case of an area of use crossing the antimeridian (longitude +/-
    /// 180 degrees), west_lon_degree will be greater than east_lon_degree.
    ///
    /// # Parameters
    ///
    /// * west_lon_degree: West longitude, in degrees. In `[-180,180]` range.
    /// * south_lat_degree: South latitude, in degrees. In `[-90,90]` range.
    /// * east_lon_degree: East longitude, in degrees. In `[-180,180]` range.
    /// * north_lat_degree: North latitude, in degrees. In `[-90,90]` range.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_area_set_bbox>
    pub fn set_bbox(
        &self,
        west_lon_degree: f64,
        south_lat_degree: f64,
        east_lon_degree: f64,
        north_lat_degree: f64,
    ) -> miette::Result<&Self> {
        if west_lon_degree > 90.0 || west_lon_degree < -90.0 {
            miette::bail!("");
        }
        unsafe {
            proj_sys::proj_area_set_bbox(
                self.ptr,
                west_lon_degree,
                south_lat_degree,
                east_lon_degree,
                north_lat_degree,
            )
        };
        Ok(self)
    }
}

impl Drop for crate::Area {
    fn drop(&mut self) { unsafe { proj_sys::proj_area_destroy(self.ptr) }; }
}
#[cfg(test)]
mod test {

    use crate::Area;

    #[test]
    fn test_set_bbox() -> miette::Result<()> {
        let area = Area::new();
        area.set_bbox(1.0, 2.0, 3.0, 4.0)?;
        Ok(())
    }
}
