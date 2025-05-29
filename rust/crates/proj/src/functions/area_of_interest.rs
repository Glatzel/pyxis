impl Default for crate::Area {
    fn default() -> Self { Self::new() }
}
///# Area of interest
impl crate::Area {
    /// # References
    ///* <https://proj.org/en/stable/development/reference/functions.html#c.proj_area_create>
    pub fn new() -> Self {
        Self {
            ptr: unsafe { proj_sys::proj_area_create() },
        }
    }
    /// # References
    ///* <https://proj.org/en/stable/development/reference/functions.html#c.proj_area_set_bbox>
    pub fn set_bbox(
        &self,
        west_lon_degree: f64,
        south_lat_degree: f64,
        east_lon_degree: f64,
        north_lat_degree: f64,
    ) -> &Self {
        unsafe {
            proj_sys::proj_area_set_bbox(
                self.ptr,
                west_lon_degree,
                south_lat_degree,
                east_lon_degree,
                north_lat_degree,
            )
        };
        self
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
        area.set_bbox(1.0, 2.0, 3.0, 4.0);
        Ok(())
    }
}
