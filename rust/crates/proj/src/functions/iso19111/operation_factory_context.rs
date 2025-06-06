use std::ptr;

use envoy::{AsVecPtr, ToCString, VecCString};

use crate::data_types::iso19111::*;
use crate::{Context, Proj, pj_obj_list_to_vec};
impl Context {
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_operation_factory_context>
    pub fn create_operation_factory_context(
        &self,
        authority: Option<&str>,
    ) -> OperationFactoryContext<'_>  {
        let authority = authority.map(|s| s.to_cstring());
        OperationFactoryContext {
            ctx: self,
            ptr: unsafe {
                proj_sys::proj_create_operation_factory_context(
                    self.ptr,
                    authority.map_or(ptr::null(), |s| s.as_ptr()),
                )
            },
        }
    }
}
impl OperationFactoryContext<'_> {
    pub fn from_context<'a>(
        ctx: &'a Context,
        authority: Option<&str>,
    ) -> OperationFactoryContext<'_> <'a> {
        ctx.create_operation_factory_context(authority)
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_desired_accuracy>
    pub fn set_desired_accuracy(&self, accuracy: f64) -> &Self {
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
    pub fn set_area_of_interest(
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
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_area_of_interest_name>
    pub fn set_area_of_interest_name(&self, area_name: &str) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_area_of_interest_name(
                self.ctx.ptr,
                self.ptr,
                area_name.to_cstring().as_ptr(),
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_crs_extent_use>
    pub fn set_crs_extent_use(&self, extent_use: CrsExtentUse) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_crs_extent_use(
                self.ctx.ptr,
                self.ptr,
                u32::from(extent_use),
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_spatial_criterion>
    pub fn set_spatial_criterion(&self, criterion: SpatialCriterion) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_spatial_criterion(
                self.ctx.ptr,
                self.ptr,
                u32::from(criterion),
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_grid_availability_use>
    pub fn set_grid_availability_use(&self, grid_availability_use: GridAvailabilityUse) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_grid_availability_use(
                self.ctx.ptr,
                self.ptr,
                u32::from(grid_availability_use),
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_use_proj_alternative_grid_names>
    pub fn set_use_proj_alternative_grid_names(&self, use_projnames: bool) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_use_proj_alternative_grid_names(
                self.ctx.ptr,
                self.ptr,
                use_projnames as i32,
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_allow_use_intermediate_crs>
    pub fn set_allow_use_intermediate_crs(
        &self,
        proj_intermediate_crs_use: IntermediateCrsUse,
    ) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_allow_use_intermediate_crs(
                self.ctx.ptr,
                self.ptr,
                u32::from(proj_intermediate_crs_use),
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_allowed_intermediate_crs>
    pub fn set_allowed_intermediate_crs(&self, list_of_auth_name_codes: &[&str]) -> &Self {
        let list_of_auth_name_codes: VecCString = list_of_auth_name_codes.into();
        unsafe {
            proj_sys::proj_operation_factory_context_set_allowed_intermediate_crs(
                self.ctx.ptr,
                self.ptr,
                list_of_auth_name_codes.as_vec_ptr().as_ptr(),
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_discard_superseded>
    pub fn set_discard_superseded(&self, discard: bool) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_discard_superseded(
                self.ctx.ptr,
                self.ptr,
                discard as i32,
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_allow_ballpark_transformations>
    pub fn set_allow_ballpark_transformations(&self, allow: bool) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_allow_ballpark_transformations(
                self.ctx.ptr,
                self.ptr,
                allow as i32,
            );
        }
        self
    }
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_operations>
    pub fn create_operations(
        &self,
        source_crs: &Proj,
        target_crs: &Proj,
    ) -> miette::Result<Vec<Proj<'_>>> {
        let ptr = unsafe {
            proj_sys::proj_create_operations(
                self.ctx.ptr,
                source_crs.ptr(),
                target_crs.ptr(),
                self.ptr,
            )
        };
        pj_obj_list_to_vec(self.ctx, ptr)
    }
}
impl Drop for OperationFactoryContext<'_> {
    fn drop(&mut self) {
        unsafe {
            proj_sys::proj_operation_factory_context_destroy(self.ptr);
        }
    }
}
#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_settings() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let factory = OperationFactoryContext::from_context(&ctx, None);
        factory
            .set_desired_accuracy(1.0)
            .set_area_of_interest(-60.0, 90.0, 60.0, 90.0)
            .set_area_of_interest_name("area_name")
            .set_crs_extent_use(CrsExtentUse::Both)
            .set_spatial_criterion(SpatialCriterion::PartialIntersection)
            .set_use_proj_alternative_grid_names(false)
            .set_allow_use_intermediate_crs(IntermediateCrsUse::Never)
            .set_discard_superseded(false)
            .set_allow_ballpark_transformations(false);

        Ok(())
    }
    #[test]
    fn test_create_operations() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let factory = OperationFactoryContext::from_context(&ctx, None);
        let source_crs = ctx.create_from_database(
            "EPSG",
            "4267",
            crate::data_types::iso19111::Category::Crs,
            false,
        )?;
        let target_crs = ctx.create_from_database("EPSG", "4269", Category::Crs, false)?;
        let ops = factory.create_operations(&source_crs, &target_crs)?;
        assert_eq!(ops.len(), 1);
        Ok(())
    }
}
