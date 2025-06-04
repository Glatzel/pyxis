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
    /// <>
    pub fn operation_factory_context_set_desired_accuracy(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_area_of_interest(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_area_of_interest_name(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_crs_extent_use(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_spatial_criterion(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_grid_availability_use(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_use_proj_alternative_grid_names(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_allow_use_intermediate_crs(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_allowed_intermediate_crs(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_discard_superseded(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn operation_factory_context_set_allow_ballpark_transformations(&self) { todo!() }
    ///# References
    ///
    /// <>
    pub fn create_operations(&self) { todo!() }
}
impl Drop for OperationFactoryContext<'_> {
    fn drop(&mut self) {
        unsafe {
            proj_sys::proj_operation_factory_context_destroy(self.ptr);
        }
    }
}
