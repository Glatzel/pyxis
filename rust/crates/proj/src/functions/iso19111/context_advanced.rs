use alloc::sync::Arc;
use core::ptr;

use crate::data_types::{ProjError, ProjErrorCode};
extern crate alloc;
use envoy::{AsVecPtr, ToCString};

use crate::data_types::iso19111::*;
use crate::{Context, OwnedCStrings, Proj, ProjOptions};
/// # ISO-19111 Advanced functions
impl Context {
    ///Instantiate a CoordinateSystem.
    ///
    /// # Arguments
    ///
    /// * `type`: Coordinate system type.
    /// * `axis`: Axis description (array of size axis_count)
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_cs>
    pub fn create_cs(
        self: &Arc<Self>,
        coordinate_system_type: CoordinateSystemType,
        axis: &[AxisDescription],
    ) -> Result<Proj, ProjError> {
        let axis_count = axis.len();
        let mut axis_vec: Vec<proj_sys::PJ_AXIS_DESCRIPTION> = Vec::with_capacity(axis_count);
        for a in axis {
            axis_vec.push(proj_sys::PJ_AXIS_DESCRIPTION {
                name: a.name.as_ptr().cast_mut(),
                abbreviation: a.abbreviation.as_ptr().cast_mut(),
                direction: a.direction.as_ptr().cast_mut(),
                unit_name: a.unit_name.as_ptr().cast_mut(),
                unit_conv_factor: a.unit_conv_factor,
                unit_type: a.unit_type.into(),
            });
        }
        let ptr = unsafe {
            proj_sys::proj_create_cs(
                self.ptr,
                coordinate_system_type.into(),
                axis_count as i32,
                axis_vec.as_ptr(),
            )
        };
        Proj::new(self, ptr)
    }
    ///Instantiate a CartesiansCS 2D.
    ///
    /// # Arguments
    ///
    /// * `type`: Coordinate system type.
    /// * `unit_name`: Unit name.
    /// * `unit_conv_factor`: Unit conversion factor to SI.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_cartesian_2D_cs>
    pub fn create_cartesian_2d_cs(
        self: &Arc<Self>,
        ellipsoidal_cs_2d_type: CartesianCs2dType,
        unit_name: Option<&str>,
        unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let unit_name = unit_name.map(|s| s.to_cstring());
        let ptr = unsafe {
            proj_sys::proj_create_cartesian_2D_cs(
                self.ptr,
                ellipsoidal_cs_2d_type.into(),
                unit_name.map_or(ptr::null(), |s| s.as_ptr()),
                unit_conv_factor,
            )
        };
        Proj::new(self, ptr)
    }
    ///Instantiate a Ellipsoidal 2D.
    ///
    /// # Arguments
    ///
    /// * `type`: Coordinate system type.
    /// * `unit_name`: Name of the angular units. Or `None` for Degree
    /// * `unit_conv_factor`: Conversion factor from the angular unit to radian.
    ///   Or 0 for Degree if unit_name == `None`. Otherwise should be not `None`
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_ellipsoidal_2D_cs>
    pub fn create_ellipsoidal_2d_cs(
        self: &Arc<Self>,
        ellipsoidal_cs_2d_type: EllipsoidalCs2dType,
        unit_name: Option<&str>,
        unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr = unsafe {
            proj_sys::proj_create_ellipsoidal_2D_cs(
                self.ptr,
                ellipsoidal_cs_2d_type.into(),
                owned.push_option(unit_name),
                unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a Ellipsoidal 3D.
    ///
    /// # Arguments
    ///
    /// * `type`: Coordinate system type.
    /// * `horizontal_angular_unit_name`: Name of the angular units. Or `None`
    ///   for Degree.
    /// * `horizontal_angular_unit_conv_factor`: Conversion factor from the
    ///   angular unit to radian. Or 0 for Degree if
    ///   horizontal_angular_unit_name == `None`. Otherwise should be not `None`
    /// * `vertical_linear_unit_name`: Vertical linear unit name. Or `None` for
    ///   Metre.
    /// * `vertical_linear_unit_conv_factor`: Vertical linear unit conversion
    ///   factor to metre. Or 0 for Metre if vertical_linear_unit_name ==
    ///   `None`. Otherwise should be not `None`
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_ellipsoidal_3D_cs>
    pub fn create_ellipsoidal_3d_cs(
        self: &Arc<Self>,
        ellipsoidal_cs_3d_type: EllipsoidalCs3dType,
        horizontal_angular_unit_name: Option<&str>,
        horizontal_angular_unit_conv_factor: f64,
        vertical_linear_unit_name: Option<&str>,
        vertical_linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let horizontal_angular_unit_name = horizontal_angular_unit_name.map(|s| s.to_cstring());
        let vertical_linear_unit_name = vertical_linear_unit_name.map(|s| s.to_cstring());
        let ptr = unsafe {
            proj_sys::proj_create_ellipsoidal_3D_cs(
                self.ptr,
                ellipsoidal_cs_3d_type.into(),
                horizontal_angular_unit_name.map_or(ptr::null(), |s| s.as_ptr()),
                horizontal_angular_unit_conv_factor,
                vertical_linear_unit_name.map_or(ptr::null(), |s| s.as_ptr()),
                vertical_linear_unit_conv_factor,
            )
        };
        Proj::new(self, ptr)
    }

    ///Create a GeographicCRS.
    ///
    /// # Arguments
    ///
    /// * `crs_name`: Name of the GeographicCRS. Or `None`
    /// * `datum_name`: Name of the GeodeticReferenceFrame. Or `None`
    /// * `ellps_name`: Name of the Ellipsoid. Or `None`
    /// * `semi_major_metre`: Ellipsoid semi-major axis, in metres.
    /// * `inv_flattening`: Ellipsoid inverse flattening. Or 0 for a sphere.
    /// * `prime_meridian_name`: Name of the PrimeMeridian. Or `None`
    /// * `prime_meridian_offset`: Offset of the prime meridian, expressed in
    ///   the specified angular units.
    /// * `pm_angular_units`: Name of the angular units. Or `None` for Degree
    /// * `pm_angular_units_conv`: Conversion factor from the angular unit to
    ///   radian. Or 0 for Degree if pm_angular_units == `None`. Otherwise
    ///   should be not `None`
    /// * `ellipsoidal_cs`: Coordinate system.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_geographic_crs>
    pub fn create_geographic_crs(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        datum_name: Option<&str>,
        ellps_name: Option<&str>,
        semi_major_metre: f64,
        inv_flattening: f64,
        prime_meridian_name: Option<&str>,
        prime_meridian_offset: f64,
        pm_angular_units: Option<&str>,
        pm_units_conv: f64,
        ellipsoidal_cs: &Proj,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(5);
        let ptr = unsafe {
            proj_sys::proj_create_geographic_crs(
                self.ptr,
                owned.push_option(crs_name),
                owned.push_option(datum_name),
                owned.push_option(ellps_name),
                semi_major_metre,
                inv_flattening,
                owned.push_option(prime_meridian_name),
                prime_meridian_offset,
                owned.push_option(pm_angular_units),
                pm_units_conv,
                ellipsoidal_cs.ptr(),
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Create a GeographicCRS.
    ///
    /// # Arguments
    ///
    /// * `crs_name`: Name of the GeographicCRS. Or `None`
    /// * `datum_or_datum_ensemble`: Datum or DatumEnsemble (DatumEnsemble
    ///   possible since 7.2).
    /// * `ellipsoidal_cs`: Coordinate system.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_geographic_crs_from_datum>
    pub fn create_geographic_crs_from_datum(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        datum_or_datum_ensemble: &Proj,
        ellipsoidal_cs: &Proj,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr = unsafe {
            proj_sys::proj_create_geographic_crs_from_datum(
                self.ptr,
                owned.push_option(crs_name),
                datum_or_datum_ensemble.ptr(),
                ellipsoidal_cs.ptr(),
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Create a GeodeticCRS of geocentric type.
    ///
    /// # Arguments
    ///
    /// * `crs_name`: Name of the GeographicCRS. Or `None`
    /// * `datum_name`: Name of the GeodeticReferenceFrame. Or `None`
    /// * `ellps_name`: Name of the Ellipsoid. Or `None`
    /// * `semi_major_metre`: Ellipsoid semi-major axis, in metres.
    /// * `inv_flattening`: Ellipsoid inverse flattening. Or 0 for a sphere.
    /// * `prime_meridian_name`: Name of the PrimeMeridian. Or `None`
    /// * `prime_meridian_offset`: Offset of the prime meridian, expressed in
    ///   the specified angular units.
    /// * `angular_units`: Name of the angular units. Or `None` for Degree
    /// * `angular_units_conv`: Conversion factor from the angular unit to
    ///   radian. Or 0 for Degree if angular_units == `None`. Otherwise should
    ///   be not `None`
    /// * `linear_units`: Name of the linear units. Or `None` for Metre
    /// * `linear_units_conv`: Conversion factor from the linear unit to metre.
    ///   Or 0 for Metre if linear_units == `None`. Otherwise should be not
    ///   `None`
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_geocentric_crs>
    pub fn create_geocentric_crs(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        datum_name: Option<&str>,
        ellps_name: Option<&str>,
        semi_major_metre: f64,
        inv_flattening: f64,
        prime_meridian_name: Option<&str>,
        prime_meridian_offset: f64,
        angular_units: Option<&str>,
        angular_units_conv: f64,
        linear_units: Option<&str>,
        linear_units_conv: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(6);
        let ptr = unsafe {
            proj_sys::proj_create_geocentric_crs(
                self.ptr,
                owned.push_option(crs_name),
                owned.push_option(datum_name),
                owned.push_option(ellps_name),
                semi_major_metre,
                inv_flattening,
                owned.push_option(prime_meridian_name),
                prime_meridian_offset,
                owned.push_option(angular_units),
                angular_units_conv,
                owned.push_option(linear_units),
                linear_units_conv,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Create a GeodeticCRS of geocentric type.
    ///
    /// # Arguments
    ///
    /// * `crs_name`: Name of the GeographicCRS. Or `None`
    /// * `datum_or_datum_ensemble`: Datum or DatumEnsemble (DatumEnsemble
    ///   possible since 7.2).
    /// * `linear_units`: Name of the linear units. Or `None` for Metre
    /// * `linear_units_conv`: Conversion factor from the linear unit to metre.
    ///   Or 0 for Metre if linear_units == `None`. Otherwise should be not
    ///   `None`
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_geocentric_crs_from_datum>
    pub fn create_geocentric_crs_from_datum(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        datum_or_datum_ensemble: &Proj,
        linear_units: Option<&str>,
        linear_units_conv: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_geocentric_crs_from_datum(
                self.ptr,
                owned.push_option(crs_name),
                datum_or_datum_ensemble.ptr(),
                owned.push_option(linear_units),
                linear_units_conv,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Create a DerivedGeograhicCRS.
    ///
    /// # Arguments
    ///
    /// * `crs_name`: Name of the GeographicCRS. Or `None`
    /// * `base_geographic_crs`: Base Geographic CRS.
    /// * `conversion`: Conversion from the base Geographic to the
    ///   DerivedGeograhicCRS.
    /// * `ellipsoidal_cs`: Coordinate system.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_derived_geographic_crs>
    pub fn create_derived_geographic_crs(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        base_geographic_crs: &Proj,
        conversion: &Proj,
        ellipsoidal_cs: &Proj,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr = unsafe {
            proj_sys::proj_create_derived_geographic_crs(
                self.ptr,
                owned.push_option(crs_name),
                base_geographic_crs.ptr(),
                conversion.ptr(),
                ellipsoidal_cs.ptr(),
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }

    ///Instantiate a EngineeringCRS with just a name.
    ///
    /// # Arguments
    ///
    /// `crs_name`: CRS name. Or `None`.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_engineering_crs>
    pub fn create_engineering_crs(
        self: &Arc<Self>,
        crs_name: Option<&str>,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr =
            unsafe { proj_sys::proj_create_engineering_crs(self.ptr, owned.push_option(crs_name)) };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a EngineeringCRS with just a name.
    ///
    ///# References
    ///
    /// # Arguments
    ///
    /// * `crs_name`: Name of the GeographicCRS. Or `None`
    /// * `datum_name`: Name of the VerticalReferenceFrame. Or `None`
    /// * `linear_units`: Name of the linear units. Or `None` for Metre
    /// * `linear_units_conv`: Conversion factor from the linear unit to metre.
    ///   Or 0 for Metre if linear_units == `None`. Otherwise should be not
    ///   `None`
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_vertical_crs>
    pub fn create_vertical_crs(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        datum_name: Option<&str>,
        linear_units: Option<&str>,
        linear_units_conv: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(3);
        let ptr = unsafe {
            proj_sys::proj_create_vertical_crs(
                self.ptr,
                owned.push_option(crs_name),
                owned.push_option(datum_name),
                owned.push_option(linear_units),
                linear_units_conv,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Create a VerticalCRS.
    ///
    /// This is an extended (_ex) version of [`Self::create_vertical_crs()`]
    /// that adds the capability of defining a geoid model.
    ///
    /// # Arguments
    ///
    /// * `crs_name`: Name of the GeographicCRS. Or `None`
    /// * `datum_name`: Name of the VerticalReferenceFrame. Or `None`
    /// * `datum_auth_name`: Authority name of the VerticalReferenceFrame. Or
    ///   `None`
    /// * `datum_code`: Code of the VerticalReferenceFrame. Or `None`
    /// * `linear_units`: Name of the linear units. Or `None` for Metre
    /// * `linear_units_conv`: Conversion factor from the linear unit to metre.
    ///   Or 0 for Metre if linear_units == `None`. Otherwise should be not
    ///   `None`
    /// * `geoid_model_name`: Geoid model name, or `None`. Can be a name from
    ///   the geoid_model name or a string "PROJ foo.gtx"
    /// * `geoid_model_auth_name`: Authority name of the transformation for the
    ///   geoid model. or `None`
    /// * `geoid_model_code`: Code of the transformation for the geoid model. or
    ///   `None`
    /// * `geoid_geog_crs`: Geographic CRS for the geoid transformation, or
    ///   `None`.
    /// * `options`: `None`-terminated list of strings with "KEY=VALUE" format.
    ///   or `None`. The currently recognized option is ACCURACY=value, where
    ///   value is in metre.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_vertical_crs_ex>
    pub fn create_vertical_crs_ex(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        datum_name: Option<&str>,
        datum_auth_name: Option<&str>,
        datum_code: Option<&str>,
        linear_units: Option<&str>,
        linear_units_conv: f64,
        geoid_model_name: Option<&str>,
        geoid_model_auth_name: Option<&str>,
        geoid_model_code: Option<&str>,
        geoid_geog_crs: Option<&Proj>,
        accuracy: Option<f64>,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(8);
        let mut options = ProjOptions::new(1);
        options.with_or_skip(accuracy, "ACCURACY");
        let ptr = unsafe {
            proj_sys::proj_create_vertical_crs_ex(
                self.ptr,
                owned.push_option(crs_name),
                owned.push_option(datum_name),
                owned.push_option(datum_auth_name),
                owned.push_option(datum_code),
                owned.push_option(linear_units),
                linear_units_conv,
                owned.push_option(geoid_model_name),
                owned.push_option(geoid_model_auth_name),
                owned.push_option(geoid_model_code),
                geoid_geog_crs.map_or(ptr::null(), |crs| crs.ptr()),
                options.as_vec_ptr().as_ptr(),
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Create a CompoundCRS.
    ///
    /// # Arguments
    ///
    /// * `crs_name`: Name of the GeographicCRS. Or `None`
    /// * `horiz_crs`: Horizontal CRS.
    /// * `vert_crs`: Vertical CRS.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_compound_crs>
    pub fn create_compound_crs(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        horiz_crs: &Proj,
        vert_crs: &Proj,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr = unsafe {
            proj_sys::proj_create_compound_crs(
                self.ptr,
                owned.push_option(crs_name),
                horiz_crs.ptr(),
                vert_crs.ptr(),
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a Conversion.
    ///
    /// # Arguments
    ///
    /// * `name`: Conversion name. Or `None`.
    /// * `auth_name`: Conversion authority name. Or `None`.
    /// * `code`: Conversion code. Or `None`.
    /// * `method_name`: Method name. Or `None`.
    /// * `method_auth_name`: Method authority name. Or `None`.
    /// * `method_code`: Method code. Or `None`.
    /// * `param_count`: Number of parameters (size of params argument)
    /// * `params`: Parameter descriptions (array of size param_count)
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion>
    pub fn create_conversion(
        self: &Arc<Self>,
        name: Option<&str>,
        auth_name: Option<&str>,
        code: Option<&str>,
        method_name: Option<&str>,
        method_auth_name: Option<&str>,
        method_code: Option<&str>,
        params: &[ParamDescription],
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(6);
        let ptr = unsafe {
            proj_sys::proj_create_conversion(
                self.ptr,
                owned.push_option(name),
                owned.push_option(auth_name),
                owned.push_option(code),
                owned.push_option(method_name),
                owned.push_option(method_auth_name),
                owned.push_option(method_code),
                params.len() as i32,
                params
                    .iter()
                    .map(|p| proj_sys::PJ_PARAM_DESCRIPTION {
                        name: p.name().to_owned().map_or(ptr::null(), |p| p.as_ptr()),
                        auth_name: p.auth_name().to_owned().map_or(ptr::null(), |p| p.as_ptr()),
                        code: p.code().to_owned().map_or(ptr::null(), |p| p.as_ptr()),
                        value: *p.value(),
                        unit_name: p.unit_name().to_owned().map_or(ptr::null(), |p| p.as_ptr()),
                        unit_conv_factor: *p.unit_conv_factor(),
                        unit_type: u32::from(*p.unit_type()),
                    })
                    .collect::<Vec<_>>()
                    .as_ptr(),
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a Transformation.
    ///
    /// # Arguments
    ///
    /// * `name`: Transformation name. Or `None`.
    /// * `auth_name`: Transformation authority name. Or `None`.
    /// * `code`: Transformation code. Or `None`.
    /// * `source_crs`: Object of type CRS representing the source CRS.
    /// * `target_crs`: Object of type CRS representing the target CRS.
    /// * `interpolation_crs`: Object of type CRS representing the interpolation
    ///   CRS. Or `None`.
    /// * `method_name`: Method name. Or `None`.
    /// * `method_auth_name`: Method authority name. Or `None`.
    /// * `method_code`: Method code. Or `None`.
    /// * `param_count`: Number of parameters (size of params argument)
    /// * `params`: Parameter descriptions (array of size param_count)
    /// * `accuracy`: Accuracy of the transformation in meters. A negative
    ///   values means unknown.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_transformation>
    pub fn create_transformation(
        self: &Arc<Self>,
        name: Option<&str>,
        auth_name: Option<&str>,
        code: Option<&str>,
        source_crs: Option<&Proj>,
        target_crs: Option<&Proj>,
        interpolation_crs: Option<&Proj>,
        method_name: Option<&str>,
        method_auth_name: Option<&str>,
        method_code: Option<&str>,
        params: &[ParamDescription],
        accuracy: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(6);
        let count = params.len();
        let params: Vec<proj_sys::PJ_PARAM_DESCRIPTION> = params
            .iter()
            .map(|p| proj_sys::PJ_PARAM_DESCRIPTION {
                name: p.name().to_owned().map_or(ptr::null(), |p| p.as_ptr()),
                auth_name: p.auth_name().to_owned().map_or(ptr::null(), |p| p.as_ptr()),
                code: p.code().to_owned().map_or(ptr::null(), |p| p.as_ptr()),
                value: *p.value(),
                unit_name: p.unit_name().to_owned().map_or(ptr::null(), |p| p.as_ptr()),
                unit_conv_factor: *p.unit_conv_factor(),
                unit_type: u32::from(*p.unit_type()),
            })
            .collect();

        let ptr = unsafe {
            proj_sys::proj_create_transformation(
                self.ptr,
                owned.push_option(name),
                owned.push_option(auth_name),
                owned.push_option(code),
                source_crs.map_or(ptr::null(), |crs| crs.ptr()),
                target_crs.map_or(ptr::null(), |crs| crs.ptr()),
                interpolation_crs.map_or(ptr::null(), |crs| crs.ptr()),
                owned.push_option(method_name),
                owned.push_option(method_auth_name),
                owned.push_option(method_code),
                count as i32,
                params.as_ptr(),
                accuracy,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Return an equivalent projection.
    ///
    /// # Arguments
    ///
    /// * `crs_name`: CRS name. Or `None`
    /// * `geodetic_crs`: Base GeodeticCRS.
    /// * `conversion`: Conversion.
    /// * `coordinate_system`: Cartesian coordinate system.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_projected_crs>
    pub fn create_projected_crs(
        self: &Arc<Self>,
        crs_name: Option<&str>,
        geodetic_crs: &Proj,
        conversion: &Proj,
        coordinate_system: &Proj,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(1);
        let ptr = unsafe {
            proj_sys::proj_create_projected_crs(
                self.ptr,
                owned.push_option(crs_name),
                geodetic_crs.ptr(),
                conversion.ptr(),
                coordinate_system.ptr(),
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Returns a BoundCRS.
    ///
    /// # Arguments
    ///
    /// * `base_crs`: Base CRS
    /// * `hub_crs`: Hub CRS
    /// * `transformation`: Transformation
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_bound_crs>
    pub fn crs_create_bound_crs(
        self: &Arc<Self>,
        base_crs: &Proj,
        hub_crs: &Proj,
        transformation: &Proj,
    ) -> Result<Proj, ProjError> {
        let ptr = unsafe {
            proj_sys::proj_crs_create_bound_crs(
                self.ptr,
                base_crs.ptr(),
                hub_crs.ptr(),
                transformation.ptr(),
            )
        };
        Proj::new(self, ptr)
    }
    ///Returns potentially a BoundCRS, with a transformation to `EPSG:4326`,
    /// wrapping this CRS.
    ///
    /// # Arguments
    ///
    /// * `vert_crs`: Object of type VerticalCRS
    /// * `hub_geographic_3D_crs`: Object of type Geographic 3D CRS
    /// * `grid_name`: Grid name (typically a .gtx file)
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_crs_create_bound_vertical_crs>
    pub fn crs_create_bound_vertical_crs(
        self: &Arc<Self>,
        vert_crs: &Proj,
        hub_geographic_3d_crs: &Proj,
        grid_name: &str,
    ) -> Result<Proj, ProjError> {
        let ptr = unsafe {
            proj_sys::proj_crs_create_bound_vertical_crs(
                self.ptr,
                vert_crs.ptr(),
                hub_geographic_3d_crs.ptr(),
                grid_name.to_cstring().as_ptr(),
            )
        };
        Proj::new(self, ptr)
    }
    ///Instantiate a ProjectedCRS with a Universal Transverse Mercator
    /// conversion.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_utm>
    pub fn create_conversion_utm(
        self: &Arc<Self>,
        zone: u8,
        north: bool,
    ) -> Result<Proj, ProjError> {
        if !(1..=60).contains(&zone) {
            return Err(ProjError {
                code: ProjErrorCode::Other,
                message: "UTM zone number should between 1 and 60.".to_string(),
            });
        }
        let ptr =
            unsafe { proj_sys::proj_create_conversion_utm(self.ptr, zone as i32, north as i32) };
        Proj::new(self, ptr)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Transverse
    /// Mercator projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_transverse_mercator>
    pub fn create_conversion_transverse_mercator(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_transverse_mercator(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Gauss
    /// Schreiber Transverse Mercator projection method.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_gauss_schreiber_transverse_mercator>
    pub fn create_conversion_gauss_schreiber_transverse_mercator(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_gauss_schreiber_transverse_mercator(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Transverse
    /// Mercator South Orientated projection method.
    ///
    /// # Arguments
    ///
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_transverse_mercator_south_oriented>
    pub fn create_conversion_transverse_mercator_south_oriented(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_transverse_mercator_south_oriented(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Two Point
    /// Equidistant projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_two_point_equidistant>
    pub fn create_conversion_two_point_equidistant(
        self: &Arc<Self>,
        latitude_first_point: f64,
        longitude_first_point: f64,
        latitude_second_point: f64,
        longitude_second_point: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_two_point_equidistant(
                self.ptr,
                latitude_first_point,
                longitude_first_point,
                latitude_second_point,
                longitude_second_point,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Tunisia Mining
    /// Grid projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_tunisia_mapping_grid>
    pub fn create_conversion_tunisia_mapping_grid(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_tunisia_mapping_grid(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Tunisia Mining
    /// Grid projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_tunisia_mining_grid>
    pub fn create_conversion_tunisia_mining_grid(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_tunisia_mining_grid(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Albers Conic
    /// Equal Area projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_albers_equal_area>
    pub fn create_conversion_albers_equal_area(
        self: &Arc<Self>,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_albers_equal_area(
                self.ptr,
                latitude_false_origin,
                longitude_false_origin,
                latitude_first_parallel,
                latitude_second_parallel,
                easting_false_origin,
                northing_false_origin,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Lambert Conic
    /// Conformal 1SP projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_1sp>
    pub fn create_conversion_lambert_conic_conformal_1sp(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_1sp(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Lambert Conic
    /// Conformal (1SP Variant B) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_1sp_variant_b>
    pub fn create_conversion_lambert_conic_conformal_1sp_variant_b(
        self: &Arc<Self>,
        latitude_nat_origin: f64,
        scale: f64,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_1sp_variant_b(
                self.ptr,
                latitude_nat_origin,
                scale,
                latitude_false_origin,
                longitude_false_origin,
                easting_false_origin,
                northing_false_origin,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Lambert Conic
    /// Conformal (2SP) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_2sp>
    pub fn create_conversion_lambert_conic_conformal_2sp(
        self: &Arc<Self>,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_2sp(
                self.ptr,
                latitude_false_origin,
                longitude_false_origin,
                latitude_first_parallel,
                latitude_second_parallel,
                easting_false_origin,
                northing_false_origin,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Lambert Conic
    /// Conformal (2SP Michigan) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_2sp_michigan>
    pub fn create_conversion_lambert_conic_conformal_2sp_michigan(
        self: &Arc<Self>,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ellipsoid_scaling_factor: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_2sp_michigan(
                self.ptr,
                latitude_false_origin,
                longitude_false_origin,
                latitude_first_parallel,
                latitude_second_parallel,
                easting_false_origin,
                northing_false_origin,
                ellipsoid_scaling_factor,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Lambert Conic
    /// Conformal (2SP Belgium) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_conic_conformal_2sp_belgium>
    pub fn create_conversion_lambert_conic_conformal_2sp_belgium(
        self: &Arc<Self>,
        latitude_false_origin: f64,
        longitude_false_origin: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        easting_false_origin: f64,
        northing_false_origin: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_conic_conformal_2sp_belgium(
                self.ptr,
                latitude_false_origin,
                longitude_false_origin,
                latitude_first_parallel,
                latitude_second_parallel,
                easting_false_origin,
                northing_false_origin,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Modified
    /// Azimuthal Equidistant projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_azimuthal_equidistant>
    pub fn create_conversion_azimuthal_equidistant(
        self: &Arc<Self>,
        latitude_nat_origin: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_azimuthal_equidistant(
                self.ptr,
                latitude_nat_origin,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Guam
    /// Projection projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_guam_projection>
    pub fn create_conversion_guam_projection(
        self: &Arc<Self>,
        latitude_nat_origin: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_guam_projection(
                self.ptr,
                latitude_nat_origin,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Bonne
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_bonne>
    pub fn create_conversion_bonne(
        self: &Arc<Self>,
        latitude_nat_origin: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_bonne(
                self.ptr,
                latitude_nat_origin,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Lambert
    /// Cylindrical Equal Area (Spherical) projection method.
    ///
    /// # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_cylindrical_equal_area_spherical>
    pub fn create_conversion_lambert_cylindrical_equal_area_spherical(
        self: &Arc<Self>,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_cylindrical_equal_area_spherical(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Lambert
    /// Cylindrical Equal Area (ellipsoidal form) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_cylindrical_equal_area>
    pub fn create_conversion_lambert_cylindrical_equal_area(
        self: &Arc<Self>,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_cylindrical_equal_area(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the
    /// Cassini-Soldner projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_cassini_soldner>
    pub fn create_conversion_cassini_soldner(
        self: &Arc<Self>,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_cassini_soldner(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Equidistant
    /// Conic projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_equidistant_conic>
    pub fn create_conversion_equidistant_conic(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_equidistant_conic(
                self.ptr,
                center_lat,
                center_long,
                latitude_first_parallel,
                latitude_second_parallel,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Eckert I
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_i>
    pub fn create_conversion_eckert_i(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_i(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Eckert II
    /// projection method. # References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_ii>
    pub fn create_conversion_eckert_ii(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_ii(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Eckert III
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_iii>
    pub fn create_conversion_eckert_iii(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_iii(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Eckert IV
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_iv>
    pub fn create_conversion_eckert_iv(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_iv(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Eckert V
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_v>
    pub fn create_conversion_eckert_v(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_v(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Eckert VI
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_eckert_vi>
    pub fn create_conversion_eckert_vi(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_eckert_vi(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Equidistant
    /// Cylindrical projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_equidistant_cylindrical>
    pub fn create_conversion_equidistant_cylindrical(
        self: &Arc<Self>,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_equidistant_cylindrical(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Equidistant
    /// Cylindrical (Spherical) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_equidistant_cylindrical_spherical>
    pub fn create_conversion_equidistant_cylindrical_spherical(
        self: &Arc<Self>,
        latitude_first_parallel: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_equidistant_cylindrical_spherical(
                self.ptr,
                latitude_first_parallel,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Gall
    /// (Stereographic) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_gall>
    pub fn create_conversion_gall(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_gall(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Goode
    /// Homolosine projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_goode_homolosine>
    pub fn create_conversion_goode_homolosine(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_goode_homolosine(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Interrupted
    /// Goode Homolosine projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_interrupted_goode_homolosine>
    pub fn create_conversion_interrupted_goode_homolosine(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_interrupted_goode_homolosine(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Geostationary
    /// Satellite View projection method, with the sweep angle axis of the
    /// viewing instrument being x.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_geostationary_satellite_sweep_x>
    pub fn create_conversion_geostationary_satellite_sweep_x(
        self: &Arc<Self>,
        center_long: f64,
        height: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_geostationary_satellite_sweep_x(
                self.ptr,
                center_long,
                height,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Geostationary
    /// Satellite View projection method, with the sweep angle axis of the
    /// viewing instrument being y.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_geostationary_satellite_sweep_y>
    pub fn create_conversion_geostationary_satellite_sweep_y(
        self: &Arc<Self>,
        center_long: f64,
        height: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_geostationary_satellite_sweep_y(
                self.ptr,
                center_long,
                height,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Gnomonic
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_gnomonic>
    pub fn create_conversion_gnomonic(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_gnomonic(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Hotine Oblique
    /// Mercator (Variant A) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_hotine_oblique_mercator_variant_a>
    pub fn create_conversion_hotine_oblique_mercator_variant_a(
        self: &Arc<Self>,
        latitude_projection_centre: f64,
        longitude_projection_centre: f64,
        azimuth_initial_line: f64,
        angle_from_rectified_to_skrew_grid: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_hotine_oblique_mercator_variant_a(
                self.ptr,
                latitude_projection_centre,
                longitude_projection_centre,
                azimuth_initial_line,
                angle_from_rectified_to_skrew_grid,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Hotine Oblique
    /// Mercator (Variant B) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_hotine_oblique_mercator_variant_b>
    pub fn create_conversion_hotine_oblique_mercator_variant_b(
        self: &Arc<Self>,
        latitude_projection_centre: f64,
        longitude_projection_centre: f64,
        azimuth_initial_line: f64,
        angle_from_rectified_to_skrew_grid: f64,
        scale: f64,
        easting_projection_centre: f64,
        northing_projection_centre: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_hotine_oblique_mercator_variant_b(
                self.ptr,
                latitude_projection_centre,
                longitude_projection_centre,
                azimuth_initial_line,
                angle_from_rectified_to_skrew_grid,
                scale,
                easting_projection_centre,
                northing_projection_centre,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    /// Instantiate a ProjectedCRS with a conversion based on the Hotine Oblique
    /// Mercator Two Point Natural Origin projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_hotine_oblique_mercator_two_point_natural_origin>
    pub fn create_conversion_hotine_oblique_mercator_two_point_natural_origin(
        self: &Arc<Self>,
        latitude_projection_centre: f64,
        latitude_point1: f64,
        longitude_point1: f64,
        latitude_point2: f64,
        longitude_point2: f64,
        scale: f64,
        easting_projection_centre: f64,
        northing_projection_centre: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_hotine_oblique_mercator_two_point_natural_origin(
                self.ptr,
                latitude_projection_centre,
                latitude_point1,
                longitude_point1,
                latitude_point2,
                longitude_point2,
                scale,
                easting_projection_centre,
                northing_projection_centre,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Laborde
    /// Oblique Mercator projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_laborde_oblique_mercator>
    pub fn create_conversion_laborde_oblique_mercator(
        self: &Arc<Self>,
        latitude_projection_centre: f64,
        longitude_projection_centre: f64,
        azimuth_initial_line: f64,
        scale: f64,
        easting_projection_centre: f64,
        northing_projection_centre: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_laborde_oblique_mercator(
                self.ptr,
                latitude_projection_centre,
                longitude_projection_centre,
                azimuth_initial_line,
                scale,
                easting_projection_centre,
                northing_projection_centre,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the International
    /// Map of the World Polyconic projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_international_map_world_polyconic>
    pub fn create_conversion_international_map_world_polyconic(
        self: &Arc<Self>,
        center_long: f64,
        latitude_first_parallel: f64,
        latitude_second_parallel: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_international_map_world_polyconic(
                self.ptr,
                center_long,
                latitude_first_parallel,
                latitude_second_parallel,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Krovak (north
    /// oriented) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_krovak_north_oriented>
    pub fn create_conversion_krovak_north_oriented(
        self: &Arc<Self>,
        latitude_projection_centre: f64,
        longitude_of_origin: f64,
        colatitude_cone_axis: f64,
        latitude_pseudo_standard_parallel: f64,
        scale_factor_pseudo_standard_parallel: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_krovak_north_oriented(
                self.ptr,
                latitude_projection_centre,
                longitude_of_origin,
                colatitude_cone_axis,
                latitude_pseudo_standard_parallel,
                scale_factor_pseudo_standard_parallel,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Krovak
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_krovak>
    pub fn create_conversion_krovak(
        self: &Arc<Self>,
        latitude_projection_centre: f64,
        longitude_of_origin: f64,
        colatitude_cone_axis: f64,
        latitude_pseudo_standard_parallel: f64,
        scale_factor_pseudo_standard_parallel: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_krovak(
                self.ptr,
                latitude_projection_centre,
                longitude_of_origin,
                colatitude_cone_axis,
                latitude_pseudo_standard_parallel,
                scale_factor_pseudo_standard_parallel,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Lambert
    /// Azimuthal Equal Area projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_lambert_azimuthal_equal_area>
    pub fn create_conversion_lambert_azimuthal_equal_area(
        self: &Arc<Self>,
        latitude_nat_origin: f64,
        longitude_nat_origin: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_lambert_azimuthal_equal_area(
                self.ptr,
                latitude_nat_origin,
                longitude_nat_origin,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Miller
    /// Cylindrical projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_miller_cylindrical>
    pub fn create_conversion_miller_cylindrical(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_miller_cylindrical(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Mercator
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_mercator_variant_a>
    pub fn create_conversion_mercator_variant_a(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_mercator_variant_a(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Mercator
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_mercator_variant_b>
    pub fn create_conversion_mercator_variant_b(
        self: &Arc<Self>,
        latitude_first_parallel: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_mercator_variant_b(
                self.ptr,
                latitude_first_parallel,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Popular
    /// Visualisation Pseudo Mercator projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_popular_visualisation_pseudo_mercator>
    pub fn create_conversion_popular_visualisation_pseudo_mercator(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_popular_visualisation_pseudo_mercator(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Mollweide
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_mollweide>
    pub fn create_conversion_mollweide(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_mollweide(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the New Zealand
    /// Map Grid projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_new_zealand_mapping_grid>
    pub fn create_conversion_new_zealand_mapping_grid(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_new_zealand_mapping_grid(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Oblique
    /// Stereographic (Alternative) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_oblique_stereographic>
    pub fn create_conversion_oblique_stereographic(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_new_zealand_mapping_grid(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Orthographic
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_orthographic>
    pub fn create_conversion_orthographic(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_orthographic(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Local
    /// Orthographic projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_local_orthographic>
    pub fn create_conversion_local_orthographic(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        azimuth: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_local_orthographic(
                self.ptr,
                center_lat,
                center_long,
                azimuth,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the American
    /// Polyconic projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_american_polyconic>
    pub fn create_conversion_american_polyconic(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_american_polyconic(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Polar
    /// Stereographic (Variant A) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_polar_stereographic_variant_a>
    pub fn create_conversion_polar_stereographic_variant_a(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_polar_stereographic_variant_a(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Polar
    /// Stereographic (Variant B) projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_polar_stereographic_variant_b>
    pub fn create_conversion_polar_stereographic_variant_b(
        self: &Arc<Self>,
        latitude_standard_parallel: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_mercator_variant_b(
                self.ptr,
                latitude_standard_parallel,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Robinson
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_robinson>
    pub fn create_conversion_robinson(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_robinson(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Sinusoidal
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_sinusoidal>
    pub fn create_conversion_sinusoidal(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_sinusoidal(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Stereographic
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_stereographic>
    pub fn create_conversion_stereographic(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        scale: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_stereographic(
                self.ptr,
                center_lat,
                center_long,
                scale,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Van der
    /// Grinten projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_van_der_grinten>
    pub fn create_conversion_van_der_grinten(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_van_der_grinten(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Wagner I
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_i>
    pub fn create_conversion_wagner_i(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_i(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Wagner II
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_ii>
    pub fn create_conversion_wagner_ii(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_ii(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Wagner III
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_iii>
    pub fn create_conversion_wagner_iii(
        self: &Arc<Self>,
        latitude_true_scale: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_iii(
                self.ptr,
                latitude_true_scale,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Wagner IV
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_iv>
    pub fn create_conversion_wagner_iv(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_iv(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Wagner V
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_v>
    pub fn create_conversion_wagner_v(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_v(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Wagner VI
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_vi>
    pub fn create_conversion_wagner_vi(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_vi(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Wagner VII
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_wagner_vii>
    pub fn create_conversion_wagner_vii(
        self: &Arc<Self>,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_wagner_vii(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the
    /// Quadrilateralized Spherical Cube projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_quadrilateralized_spherical_cube>
    pub fn create_conversion_quadrilateralized_spherical_cube(
        self: &Arc<Self>,
        center_lat: f64,
        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_quadrilateralized_spherical_cube(
                self.ptr,
                center_lat,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Spherical
    /// Cross-Track Height projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_spherical_cross_track_height>
    pub fn create_conversion_spherical_cross_track_height(
        self: &Arc<Self>,
        peg_point_lat: f64,
        peg_point_long: f64,
        peg_point_heading: f64,
        peg_point_height: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_spherical_cross_track_height(
                self.ptr,
                peg_point_lat,
                peg_point_long,
                peg_point_heading,
                peg_point_height,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a ProjectedCRS with a conversion based on the Equal Earth
    /// projection method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_equal_earth>
    pub fn create_conversion_equal_earth(
        self: &Arc<Self>,

        center_long: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_equal_earth(
                self.ptr,
                center_long,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a conversion based on the Vertical Perspective projection
    /// method.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_vertical_perspective>
    pub fn create_conversion_vertical_perspective(
        self: &Arc<Self>,
        topo_origin_lat: f64,
        topo_origin_long: f64,
        topo_origin_height: f64,
        view_point_height: f64,
        false_easting: f64,
        false_northing: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
        linear_unit_name: Option<&str>,
        linear_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_vertical_perspective(
                self.ptr,
                topo_origin_lat,
                topo_origin_long,
                topo_origin_height,
                view_point_height,
                false_easting,
                false_northing,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
                owned.push_option(linear_unit_name),
                linear_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a conversion based on the Pole Rotation method, using the
    /// conventions of the GRIB 1 and GRIB 2 data formats.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_pole_rotation_grib_convention>
    pub fn create_conversion_pole_rotation_grib_convention(
        self: &Arc<Self>,
        south_pole_lat_in_unrotated_crs: f64,
        south_pole_long_in_unrotated_crs: f64,
        axis_rotation: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_pole_rotation_grib_convention(
                self.ptr,
                south_pole_lat_in_unrotated_crs,
                south_pole_long_in_unrotated_crs,
                axis_rotation,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
    ///Instantiate a conversion based on the Pole Rotation method, using the
    /// conventions of the netCDF CF convention for the netCDF format.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_conversion_pole_rotation_netcdf_cf_convention>
    pub fn create_conversion_pole_rotation_netcdf_cf_convention(
        self: &Arc<Self>,
        grid_north_pole_latitude: f64,
        grid_north_pole_longitude: f64,
        north_pole_grid_longitude: f64,
        ang_unit_name: Option<&str>,
        ang_unit_conv_factor: f64,
    ) -> Result<Proj, ProjError> {
        let mut owned = OwnedCStrings::with_capacity(2);
        let ptr = unsafe {
            proj_sys::proj_create_conversion_pole_rotation_netcdf_cf_convention(
                self.ptr,
                grid_north_pole_latitude,
                grid_north_pole_longitude,
                north_pole_grid_longitude,
                owned.push_option(ang_unit_name),
                ang_unit_conv_factor,
            )
        };
        Proj::new_with_owned_cstrings(self, ptr, owned)
    }
}

#[cfg(test)]
mod test_context_advanced {
    use strum::IntoEnumIterator;

    use super::*;
    #[test]
    fn test_create_cs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        for a in AxisDirection::iter() {
            let pj: Proj = ctx.create_cs(
                CoordinateSystemType::Cartesian,
                &[
                    AxisDescription::new(
                        Some("Longitude"),
                        Some("lon"),
                        a,
                        Some("Degree"),
                        1.0,
                        UnitType::Angular,
                    )?,
                    AxisDescription::new(
                        Some("Latitude"),
                        Some("lat"),
                        AxisDirection::North,
                        Some("Degree"),
                        1.0,
                        UnitType::Angular,
                    )?,
                ],
            )?;
            let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
            println!("{wkt}\n");
            assert!(wkt.contains("9122"));
        }
        Ok(())
    }
    #[test]
    fn test_create_cartesian_2d_cs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj =
            ctx.create_cartesian_2d_cs(CartesianCs2dType::EastingNorthing, Some("Degree"), 1.0)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("CS[Cartesian,2]"));
        Ok(())
    }
    #[test]
    fn test_create_ellipsoidal_2d_cs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj = ctx.create_ellipsoidal_2d_cs(
            EllipsoidalCs2dType::LatitudeLongitude,
            Some("Degree"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("CS[ellipsoidal,2]"));
        Ok(())
    }
    #[test]
    fn test_create_ellipsoidal_3d_cs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_ellipsoidal_3d_cs(
            EllipsoidalCs3dType::LatitudeLongitudeHeight,
            Some("Degree"),
            1.0,
            Some("Degree"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("CS[ellipsoidal,3]"));
        Ok(())
    }

    #[test]
    fn test_create_geographic_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj = ctx.create_geographic_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            1.0,
            Some("Degree"),
            1.0,
            &ctx.create_ellipsoidal_2d_cs(
                EllipsoidalCs2dType::LatitudeLongitude,
                Some("Degree"),
                1.0,
            )?,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("WGS 84"));
        assert!(wkt.contains("World Geodetic System 1984"));
        assert!(wkt.contains("Greenwich"));
        Ok(())
    }
    #[test]
    fn test_create_geographic_crs_from_datum() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj = ctx.create_geographic_crs_from_datum(
            Some("WGS 84"),
            &ctx.create("+proj=geocent +ellps=GRS80 +units=m +no_defs +type=crs")?
                .crs_get_datum()?
                .unwrap(),
            &ctx.create_ellipsoidal_2d_cs(
                EllipsoidalCs2dType::LatitudeLongitude,
                Some("Degree"),
                1.0,
            )?,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("GRS 1980"));
        Ok(())
    }
    #[test]
    fn test_create_create_geocentric_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj = ctx.create_geocentric_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("MyDegree"),
            0.0174532925199433,
            Some("MyMetre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("WGS 84"));
        assert!(wkt.contains("World Geodetic System 1984"));
        assert!(wkt.contains("Greenwich"));
        assert!(wkt.contains("MyDegree"));
        assert!(wkt.contains("World Geodetic System 1984"));
        assert!(wkt.contains("MyMetre"));

        Ok(())
    }
    #[test]
    fn test_create_geocentric_crs_from_datum() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj1: Proj = ctx.create_geocentric_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("MyDegree"),
            0.0174532925199433,
            Some("MyMetre1"),
            1.1,
        )?;
        let pj2: Proj = ctx.create_geocentric_crs_from_datum(
            Some("new crs"),
            &pj1.crs_get_datum()?.unwrap(),
            Some("MyMetre2"),
            1.0,
        )?;
        let wkt = pj2.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("new crs"));
        assert!(wkt.contains("MyMetre2"));
        Ok(())
    }
    #[test]
    fn test_create_derived_geographic_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let crs_4326 = ctx.create("EPSG:4326")?;
        let conversion = ctx
            .clone()
            .create_conversion_pole_rotation_grib_convention(
                2.0,
                3.0,
                4.0,
                Some("Degree"),
                0.0174532925199433,
            )?;
        let cs = crs_4326.crs_get_coordinate_system()?;
        let pj: Proj =
            ctx.create_derived_geographic_crs(Some("my rotated CRS"), &crs_4326, &conversion, &cs)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("my rotated CRS"));
        Ok(())
    }

    #[test]
    fn test_create_engineering_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;

        let pj: Proj = ctx.create_engineering_crs(Some("engineering crs"))?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("engineering crs"));
        Ok(())
    }
    #[test]
    fn test_create_vertical_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj =
            ctx.create_vertical_crs(Some("myVertCRS"), Some("myVertDatum"), None, 0.0)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("myVertDatum"));
        Ok(())
    }
    #[test]
    fn test_create_vertical_crs_ex() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj = ctx.create_vertical_crs_ex(
            Some("myVertCRS (ftUS)"),
            Some("myVertDatum"),
            None,
            None,
            Some("US survey foot"),
            0.304800609601219,
            Some("PROJ @foo.gtx"),
            None,
            None,
            None,
            None,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("myVertCRS (ftUS)"));
        Ok(())
    }
    #[test]
    fn test_create_compound_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let horiz_crs = ctx
            .clone()
            .create_from_database("EPSG", "6340", Category::Crs, false)?;
        let vert_crs: Proj = ctx.create_vertical_crs_ex(
            Some("myVertCRS (ftUS)"),
            Some("myVertDatum"),
            None,
            None,
            Some("US survey foot"),
            0.304800609601219,
            Some("PROJ @foo.gtx"),
            None,
            None,
            None,
            None,
        )?;
        let pj: Proj = ctx.create_compound_crs(Some("Compound"), &horiz_crs, &vert_crs)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("Compound"));
        Ok(())
    }
    #[test]
    fn test_create_conversion() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj: Proj = ctx.create_conversion(
            Some("conv"),
            Some("conv auth"),
            Some("conv code"),
            Some("method"),
            Some("method auth"),
            Some("method code"),
            &[ParamDescription::new(
                Some("param name".to_cstring()),
                None,
                None,
                0.99,
                None,
                1.0,
                UnitType::Scale,
            )],
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("conv"));
        Ok(())
    }
    #[test]
    fn test_create_transformation() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let geog_cs =
            ctx.create_ellipsoidal_2d_cs(EllipsoidalCs2dType::LongitudeLatitude, None, 0.0)?;
        let source_crs = ctx.create_geographic_crs(
            Some("Source CRS"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("Degree"),
            0.0174532925199433,
            &geog_cs,
        )?;
        let target_crs = ctx.create_geographic_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("Degree"),
            0.0174532925199433,
            &geog_cs,
        )?;
        let pj = ctx.create_transformation(
            Some("transf"),
            Some("transf auth"),
            Some("conv code"),
            Some(&source_crs),
            Some(&target_crs),
            Some(&target_crs),
            Some("method"),
            Some("method auth"),
            Some("method code"),
            &[ParamDescription::new(
                Some("param name".to_cstring()),
                None,
                None,
                0.99,
                None,
                1.0,
                UnitType::Scale,
            )],
            0.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("transf"));
        Ok(())
    }
    #[test]
    fn test_projected_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let conv = ctx.create_conversion(
            Some("conv"),
            Some("conv auth"),
            Some("conv code"),
            Some("method"),
            Some("method auth"),
            Some("method code"),
            &[ParamDescription::new(
                Some("param name".to_cstring()),
                None,
                None,
                0.99,
                None,
                1.0,
                UnitType::Scale,
            )],
        )?;
        let geog_cs =
            ctx.create_ellipsoidal_2d_cs(EllipsoidalCs2dType::LongitudeLatitude, None, 0.0)?;

        let geog_crs = ctx.create_geographic_crs(
            Some("WGS 84"),
            Some("World Geodetic System 1984"),
            Some("WGS 84"),
            6378137.0,
            298.257223563,
            Some("Greenwich"),
            0.0,
            Some("Degree"),
            0.0174532925199433,
            &geog_cs,
        )?;
        let cs = ctx.create_cartesian_2d_cs(CartesianCs2dType::EastingNorthing, None, 0.0)?;
        let pj: Proj = ctx.create_projected_crs(Some("my CRS"), &geog_crs, &conv, &cs)?;

        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("my CRS"));
        Ok(())
    }
    #[test]
    fn test_crs_create_bound_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let crs = ctx
            .clone()
            .create_from_database("EPSG", "4807", Category::Crs, false)?;
        let res = crs.crs_create_bound_crs_to_wgs84(None)?;
        let base_crs = res.get_source_crs().unwrap();
        let hub_crs = res.get_target_crs().unwrap();
        let transf = res.crs_get_coordoperation()?;
        let bound_crs = ctx.crs_create_bound_crs(&base_crs, &hub_crs, &transf)?;
        let wkt = bound_crs.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_crs_create_bound_vertical_crs() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let crs = ctx.create("EPSG:4979")?;
        let vert_crs =
            ctx.create_vertical_crs(Some("myVertCRS"), Some("myVertDatum"), None, 0.0)?;

        let bound_crs = ctx.crs_create_bound_vertical_crs(&vert_crs, &crs, "foo.gtx")?;
        let wkt = bound_crs.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_utm() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_utm(31, true)?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        assert!(wkt.contains("UTM zone 31N"));
        Ok(())
    }
    #[test]
    fn test_create_conversion_transverse_mercator() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_transverse_mercator(
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_gauss_schreiber_transverse_mercator() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_gauss_schreiber_transverse_mercator(
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_transverse_mercator_south_oriented() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_transverse_mercator_south_oriented(
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_two_point_equidistant() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_two_point_equidistant(
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_tunisia_mapping_grid() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_tunisia_mapping_grid(
            0.0,
            0.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_tunisia_mining_grid() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_tunisia_mining_grid(
            0.0,
            0.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_albers_equal_area() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_albers_equal_area(
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_1sp() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_1sp(
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_1sp_variant_b() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_1sp_variant_b(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_2sp() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_2sp(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_2sp_michigan() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_2sp_michigan(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_conic_conformal_2sp_belgium() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_conic_conformal_2sp_belgium(
            1.0,
            1.0,
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_azimuthal_equidistant() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_azimuthal_equidistant(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_guam_projection() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_guam_projection(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_createconversion_bonne() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_bonne(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_cylindrical_equal_area_spherical() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_cylindrical_equal_area_spherical(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_cylindrical_equal_area() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_cylindrical_equal_area(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_cassini_soldner() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_cassini_soldner(
            1.0,
            1.0,
            0.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_equidistant_conic() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_equidistant_conic(
            1.0,
            1.0,
            0.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_i() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_i(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_ii() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_ii(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_iii() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_iii(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_iv() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_iv(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_v() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_v(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_eckert_vi() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_eckert_vi(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_equidistant_cylindrical() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_equidistant_cylindrical(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_equidistant_cylindrical_spherical() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_equidistant_cylindrical_spherical(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_gall() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_gall(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_goode_homolosine() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_goode_homolosine(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_interrupted_goode_homolosine() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_interrupted_goode_homolosine(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_geostationary_satellite_sweep_x() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_geostationary_satellite_sweep_x(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_geostationary_satellite_sweep_y() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_geostationary_satellite_sweep_y(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_gnomonic() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_gnomonic(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_hotine_oblique_mercator_variant_a() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_hotine_oblique_mercator_variant_a(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_hotine_oblique_mercator_variant_b() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_hotine_oblique_mercator_variant_b(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_hotine_oblique_mercator_two_point_natural_origin()
    -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_hotine_oblique_mercator_two_point_natural_origin(
            1.0,
            1.0,
            1.0,
            2.0,
            3.0,
            1.0,
            4.0,
            6.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_laborde_oblique_mercator() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_laborde_oblique_mercator(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_international_map_world_polyconic() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_international_map_world_polyconic(
            1.0,
            2.0,
            4.0,
            5.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_krovak_north_oriented() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_krovak_north_oriented(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_krovak() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_krovak(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_lambert_azimuthal_equal_area() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_lambert_azimuthal_equal_area(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_miller_cylindrical() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_miller_cylindrical(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_mercator_variant_a() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_mercator_variant_a(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_mercator_variant_b() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_mercator_variant_b(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_popular_visualisation_pseudo_mercator() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_popular_visualisation_pseudo_mercator(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_mollweide() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_mollweide(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_new_zealand_mapping_grid() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_new_zealand_mapping_grid(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_oblique_stereographic() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_oblique_stereographic(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_orthographic() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_orthographic(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_local_orthographic() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_local_orthographic(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_american_polyconic() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_american_polyconic(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_polar_stereographic_variant_a() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_polar_stereographic_variant_a(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_polar_stereographic_variant_b() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_polar_stereographic_variant_b(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_robinson() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_robinson(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_sinusoidal() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_sinusoidal(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_stereographic() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_stereographic(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_van_der_grinten() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_van_der_grinten(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_i() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_i(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_ii() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_ii(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_iii() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_iii(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_iv() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_iv(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_v() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_v(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_vi() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_vi(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_wagner_vii() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_wagner_vii(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_quadrilateralized_spherical_cube() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_quadrilateralized_spherical_cube(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_spherical_cross_track_height() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_spherical_cross_track_height(
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }

    #[test]
    fn test_create_conversion_equal_earth() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_equal_earth(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_vertical_perspective() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_vertical_perspective(
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
            Some("Metre"),
            1.0,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_pole_rotation_grib_convention() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_pole_rotation_grib_convention(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
    #[test]
    fn test_create_conversion_pole_rotation_netcdf_cf_convention() -> Result<(), ProjError> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create_conversion_pole_rotation_netcdf_cf_convention(
            1.0,
            1.0,
            1.0,
            Some("Degree"),
            0.0174532925199433,
        )?;
        let wkt = pj.as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?;
        println!("{wkt}");
        Ok(())
    }
}
