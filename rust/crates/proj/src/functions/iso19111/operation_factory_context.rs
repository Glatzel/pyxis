use envoy::ToCStr;

use crate::Context;
use crate::data_types::iso19111::*;
impl Context {
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_operation_factory_context>
    pub fn create_operation_factory_context(
        &self,
        authority: Option<&str>,
    ) -> OperationFactoryContext {
        OperationFactoryContext {
            ctx: self,
            ptr: unsafe {
                proj_sys::proj_create_operation_factory_context(self.ptr, authority.to_cstr())
            },
        }
    }
}
impl OperationFactoryContext<'_> {
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_desired_accuracy>
    pub fn operation_factory_context_set_desired_accuracy(&self, accuracy: f64) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_desired_accuracy(
                self.ctx.ptr,
                self.ptr,
                accuracy,
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_area_of_interest>
    pub fn operation_factory_context_set_area_of_interest(
        &self,
        west_lon_degree: f64,
        south_lat_degree: f64,
        east_lon_degree: f64,
        north_lat_degree: f64,
    ) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_area_of_interest(
                self.ctx.ptr,
                self.ptr,
                west_lon_degree,
                south_lat_degree,
                east_lon_degree,
                north_lat_degree,
            );
        }
        self
    }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_area_of_interest_name(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_crs_extent_use(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_spatial_criterion(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_grid_availability_use(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_use_proj_alternative_grid_names(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_allow_use_intermediate_crs(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_allowed_intermediate_crs(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_discard_superseded(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_allow_ballpark_transformations(&self) -> &Self { todo!() }
    ///# References
    ///
    /// <>
    pub fn create_operations(&self) -> &Self { todo!() }
}
impl Drop for OperationFactoryContext<'_> {
    fn drop(&mut self) {
        unsafe {
            proj_sys::proj_operation_factory_context_destroy(self.ptr);
        }
    }
}
