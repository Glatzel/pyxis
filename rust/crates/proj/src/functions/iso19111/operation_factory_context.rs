use core::ptr;
extern crate alloc;
use envoy::{AsVecPtr, ToCString, ToVecCString};

use crate::data_types::ProjError;
use crate::data_types::iso19111::*;
use crate::{Context, Proj};
impl Context {
    ///Instantiate a context for building coordinate operations between two
    /// CRS.
    ///
    /// If authority is NULL or the empty string, then coordinate operations
    /// from any authority will be searched, with the restrictions set in the
    /// authority_to_authority_preference database table. If authority is set to
    /// "any", then coordinate operations from any authority will be searched If
    /// authority is a non-empty string different of "any", then coordinate
    /// operations will be searched only in that authority namespace.
    ///
    /// # Arguments
    ///
    /// * `authority`: Name of authority to which to restrict the search of
    ///   candidate operations.
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_operation_factory_context>
    pub fn create_operation_factory_context(
        &self,
        authority: Option<&str>,
    ) -> Result<OperationFactoryContext, ProjError> {
        let authority = authority.map(|s| s.to_cstring()).transpose()?;
        Ok(OperationFactoryContext {
            arc_ctx_ptr: self.arc_ptr(),
            ptr: unsafe {
                proj_sys::proj_create_operation_factory_context(
                    self.ptr(),
                    authority.map_or(ptr::null(), |s| s.as_ptr()),
                )
            },
        })
    }
}
impl OperationFactoryContext {
    ///# See Also
    /// * [`crate::Context::create_operation_factory_context`]
    pub fn from_context(
        ctx: &Context,
        authority: Option<&str>,
    ) -> Result<OperationFactoryContext, ProjError> {
        ctx.create_operation_factory_context(authority)
    }
    ///Set the desired accuracy of the resulting coordinate transformations.
    ///
    /// # Arguments
    ///
    /// * `accuracy`: Accuracy in meter (or 0 to disable the filter).
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_desired_accuracy>
    pub fn set_desired_accuracy(&self, accuracy: f64) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_desired_accuracy(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                accuracy,
            );
        }
        self
    }
    ///Set the desired area of interest for the resulting coordinate
    /// transformations.
    ///
    ///For an area of interest crossing the anti-meridian, west_lon_degree will
    /// be greater than east_lon_degree.
    ///
    /// # Arguments
    ///
    /// * `west_lon_degree`: West longitude (in degrees).
    /// * `south_lat_degree`: South latitude (in degrees).
    /// * `east_lon_degree`: East longitude (in degrees).
    /// * `north_lat_degree`: North latitude (in degrees).
    ///
    /// # References
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
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                west_lon_degree,
                south_lat_degree,
                east_lon_degree,
                north_lat_degree,
            );
        }
        self
    }
    ///Set the name of the desired area of interest for the resulting
    /// coordinate transformations.
    ///
    /// # Parameters
    ///
    /// * `area_name`: Area name. Must be known of the database.
    ///
    ///  # References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_area_of_interest_name>
    pub fn set_area_of_interest_name(&self, area_name: &str) -> Result<&Self, ProjError> {
        unsafe {
            proj_sys::proj_operation_factory_context_set_area_of_interest_name(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                area_name.to_cstring()?.as_ptr(),
            );
        }
        Ok(self)
    }
    ///Set how source and target CRS extent should be used when considering if
    /// a transformation can be used (only takes effect if no area of interest
    /// is explicitly defined).
    ///
    ///The default is [`CrsExtentUse::Smallest`].
    ///
    /// # Parameters
    ///
    /// * `use`: How source and target CRS extent should be used.
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_crs_extent_use>
    pub fn set_crs_extent_use(&self, extent_use: CrsExtentUse) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_crs_extent_use(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                extent_use as u32,
            );
        }
        self
    }
    ///Set the spatial criterion to use when comparing the area of validity of
    /// coordinate operations with the area of interest / area of validity of
    /// source and target CRS.
    ///
    /// The default is [`SpatialCriterion::StrictContainment`].
    ///
    /// # Arguments
    ///
    /// * `criterion`: spatial criterion to use
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_spatial_criterion>
    pub fn set_spatial_criterion(&self, criterion: SpatialCriterion) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_spatial_criterion(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                criterion as u32,
            );
        }
        self
    }
    ///Set how grid availability is used.
    ///
    /// The default is [`GridAvailabilityUse::UsedForSorting`].
    ///
    /// # Arguments
    ///
    /// \* `use`: how grid availability is used.
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_grid_availability_use>
    pub fn set_grid_availability_use(&self, grid_availability_use: GridAvailabilityUse) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_grid_availability_use(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                grid_availability_use as u32,
            );
        }
        self
    }
    /// Set whether PROJ alternative grid names should be substituted to the
    /// official authority names.
    ///
    /// The default is true.
    ///
    /// # Arguments
    ///
    /// * `use_proj_names`: whether PROJ alternative grid names should be used
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_use_proj_alternative_grid_names>
    pub fn set_use_proj_alternative_grid_names(&self, use_proj_names: bool) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_use_proj_alternative_grid_names(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                use_proj_names as i32,
            );
        }
        self
    }
    ///Set whether an intermediate pivot CRS can be used for researching
    /// coordinate operations between a source and target CRS.
    ///
    ///Concretely if in the database there is an operation from A to C (or C to
    /// A), and another one from C to B (or B to C), but no direct operation
    /// between A and B, setting this parameter to true, allow chaining both
    /// operations.
    ///
    ///The current implementation is limited to researching one intermediate
    /// step.
    ///
    ///By default, with the IF_NO_DIRECT_TRANSFORMATION strategy, all potential
    /// C candidates will be used if there is no direct transformation.
    ///
    /// # Arguments
    ///
    /// * `use`: whether and how intermediate CRS may be used.
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_allow_use_intermediate_crs>
    pub fn set_allow_use_intermediate_crs(
        &self,
        proj_intermediate_crs_use: IntermediateCrsUse,
    ) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_allow_use_intermediate_crs(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                proj_intermediate_crs_use as u32,
            );
        }
        self
    }

    ///Restrict the potential pivot CRSs that can be used when trying to build
    /// a coordinate operation between two CRS that have no direct operation.
    ///
    /// # Arguments
    ///
    /// * `list_of_auth_name_codes`: an array of strings NLL terminated, with
    ///   the format { "auth_name1", "code1", "auth_name2", "code2", ... NULL }
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_allowed_intermediate_crs>
    pub fn set_allowed_intermediate_crs(
        &self,
        list_of_auth_name_codes: &[&str],
    ) -> Result<&Self, ProjError> {
        let list_of_auth_name_codes = list_of_auth_name_codes.to_vec_cstring()?;
        unsafe {
            proj_sys::proj_operation_factory_context_set_allowed_intermediate_crs(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                list_of_auth_name_codes.as_vec_ptr().as_ptr(),
            );
        }
        Ok(self)
    }
    /// Set whether transformations that are superseded (but not deprecated)
    /// should be discarded.
    ///
    /// # Arguments
    ///
    /// * `discard`: superseded crs or not
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_discard_superseded>
    pub fn set_discard_superseded(&self, discard: bool) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_discard_superseded(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                discard as i32,
            );
        }
        self
    }
    ///Set whether ballpark transformations are allowed.
    ///
    /// # Arguments
    ///
    /// * `allow`: set to TRUE to allow ballpark transformations.
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_set_allow_ballpark_transformations>
    pub fn set_allow_ballpark_transformations(&self, allow: bool) -> &Self {
        unsafe {
            proj_sys::proj_operation_factory_context_set_allow_ballpark_transformations(
                self.arc_ctx_ptr.ptr(),
                self.ptr,
                allow as i32,
            );
        }
        self
    }
    ///Find a list of CoordinateOperation from source_crs to target_crs.
    ///
    ///The operations are sorted with the most relevant ones first: by
    /// descending area (intersection of the transformation area with the area
    /// of interest, or intersection of the transformation with the area of use
    /// of the CRS), and by increasing accuracy. Operations with unknown
    /// accuracy are sorted last, whatever their area.
    ///
    ///Starting with PROJ 9.1, vertical transformations are only done if both
    /// source CRS and target CRS are 3D CRS or Compound CRS with a vertical
    /// component. You may need to use [`Proj::crs_promote_to_3d()`].
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_operations>
    pub fn create_operations(
        &self,
        source_crs: &Proj,
        target_crs: &Proj,
    ) -> Result<ProjObjList, ProjError> {
        let ptr = unsafe {
            proj_sys::proj_create_operations(
                self.arc_ctx_ptr.ptr(),
                source_crs.ptr(),
                target_crs.ptr(),
                self.ptr,
            )
        };
        ProjObjList::new(self.arc_ctx_ptr.clone(), ptr)
    }
}
impl Drop for OperationFactoryContext {
    ///Drops a reference on an object.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_operation_factory_context_destroy>
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
    fn test_settings() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let factory = OperationFactoryContext::from_context(&ctx, None)?;
        factory
            .set_desired_accuracy(1.0)
            .set_area_of_interest(-60.0, 90.0, 60.0, 90.0)
            .set_area_of_interest_name("area_name")?
            .set_crs_extent_use(CrsExtentUse::Both)
            .set_spatial_criterion(SpatialCriterion::PartialIntersection)
            .set_use_proj_alternative_grid_names(false)
            .set_allow_use_intermediate_crs(IntermediateCrsUse::Never)
            .set_discard_superseded(false)
            .set_allow_ballpark_transformations(false);

        Ok(())
    }
    #[test]
    fn test_create_operations() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let factory = OperationFactoryContext::from_context(&ctx, None)?;
        let source_crs = ctx.create_from_database(
            "EPSG",
            "4267",
            crate::data_types::iso19111::Category::Crs,
            false,
        )?;
        let target_crs = ctx.create_from_database("EPSG", "4269", Category::Crs, false)?;
        let ops = factory.create_operations(&source_crs, &target_crs)?;
        println!(
            "{}",
            ops.get(0)?
                .as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?
        );

        Ok(())
    }
}
