impl Default for crate::PjArea {
    fn default() -> Self { Self::new() }
}
///# Area of interest
impl crate::PjArea {
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_area_create>
    pub fn new() -> Self {
        Self {
            area: unsafe { proj_sys::proj_area_create() },
        }
    }
    /// # References
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_area_set_bbox>
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

impl Drop for crate::PjArea {
    fn drop(&mut self) { unsafe { proj_sys::proj_area_destroy(self.area) }; }
}
#[cfg(test)]
mod test {
    use float_cmp::assert_approx_eq;

    use crate::PjArea;

    #[test]
    fn test_set_bbox() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let area = PjArea::new();
       area.set_bbox(1, 2, 3, 4.0);
        Ok(())
    }
}
