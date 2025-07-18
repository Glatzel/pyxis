use std::ptr;
use std::sync::Arc;

use envoy::ToCString;

use crate::data_types::iso19111::*;
use crate::{Context, OwnedCStrings, Proj, ToCoord};
impl ProjObjList {
    ///Return the index of the operation that would be the most appropriate to
    /// transform the specified coordinates.
    ///
    /// # Arguments
    ///
    /// * `direction`: Direction into which to transform the point.
    /// * `coord`: Coordinate to transform
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_suggested_operation>
    pub fn get_suggested_operation(
        &self,
        direction: crate::Direction,
        coord: impl crate::ICoord,
    ) -> miette::Result<Option<Proj>> {
        let index = unsafe {
            proj_sys::proj_get_suggested_operation(
                self.ctx.ptr,
                self.ptr(),
                i32::from(direction),
                coord.to_coord()?,
            )
        };
        if index == -1 {
            return Ok(None);
        }
        let ptr = unsafe { proj_sys::proj_list_get(self.ctx.ptr, self.ptr(), index) };

        if self._owned_cstrings.len() > 0 {
            Ok(Some(Proj::new(&self.ctx, ptr)?))
        } else {
            Ok(Some(Proj::new_with_owned_cstrings(
                &self.ctx,
                ptr,
                self._owned_cstrings.clone(),
            )?))
        }
    }
}
impl Drop for ProjObjList {
    ///Drops a reference on the result set.
    ///
    /// # References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_destroy>
    fn drop(&mut self) {
        unsafe {
            proj_sys::proj_list_destroy(self.ptr());
        }
    }
}

impl Context {
    ///Return a list of objects by their name.
    ///
    /// # Arguments
    /// * `auth_name`: Authority name, used to restrict the search. Or `None`
    ///   for all authorities.
    /// * `searched_name`: Searched name. Must be at least 2 character long.
    /// * `types`: List of object types into which to search. If `None`, all
    ///   object types will be searched.
    /// * `approximate_match`: Whether approximate name identification is
    ///   allowed.
    /// * `limit_result_count`: Maximum number of results to return. Or 0 for
    ///   unlimited.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_from_name>
    pub fn create_from_name(
        self: &Arc<Self>,
        auth_name: Option<&str>,
        searched_name: &str,
        types: Option<&[ProjType]>,
        approximate_match: bool,
        limit_result_count: usize,
    ) -> miette::Result<ProjObjList> {
        let (types, count) = types.map_or((None, 0), |types| {
            let types: Vec<u32> = types.iter().map(|f| u32::from(f.clone())).collect();
            let count = types.len();
            (Some(types), count)
        });
        let auth_name = auth_name.map(|s| s.to_cstring());
        let result = unsafe {
            proj_sys::proj_create_from_name(
                self.ptr,
                auth_name.map_or(ptr::null(), |s| s.as_ptr()),
                searched_name.to_cstring().as_ptr(),
                types.map_or(ptr::null(), |types| types.as_ptr()),
                count,
                approximate_match as i32,
                limit_result_count,
                ptr::null(),
            )
        };
        ProjObjList::new(self, result)
    }
    /// Return GeodeticCRS that use the specified datum.
    ///
    /// # Arguments
    ///
    /// * `crs_auth_name`: CRS authority name, or `None`.
    /// * `datum_auth_name`: Datum authority name
    /// * `datum_code`: Datum code
    /// * `crs_type`: "geographic 2D", "geographic 3D", "geocentric" or `None`
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_query_geodetic_crs_from_datum>
    pub fn query_geodetic_crs_from_datum(
        self: &Arc<Self>,
        crs_auth_name: Option<&str>,
        datum_auth_name: &str,
        datum_code: &str,
        crs_type: Option<&str>,
    ) -> miette::Result<ProjObjList> {
        let mut owned = OwnedCStrings::new();
        let ptr = unsafe {
            proj_sys::proj_query_geodetic_crs_from_datum(
                self.ptr,
                owned.push_option(crs_auth_name),
                datum_auth_name.to_cstring().as_ptr(),
                datum_code.to_cstring().as_ptr(),
                owned.push_option(crs_type),
            )
        };
        ProjObjList::new_with_owned_cstrings(self, ptr, owned)
    }
}
impl Proj {
    ///Return a list of non-deprecated objects related to the passed one.
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_non_deprecated>
    pub fn get_non_deprecated(&self) -> miette::Result<ProjObjList> {
        let result = unsafe { proj_sys::proj_get_non_deprecated(self.ctx.ptr, self.ptr()) };
        ProjObjList::new(&self.ctx, result)
    }

    ///Identify the CRS with reference CRSs.
    ///
    ///The candidate CRSs are either hard-coded, or looked in the database when
    /// it is available.
    ///
    ///Note that the implementation uses a set of heuristics to have a good
    /// compromise of successful identifications over execution time. It might
    /// miss legitimate matches in some circumstances.
    ///
    ///The method returns a list of matching reference CRS, and the percentage
    /// (0-100) of confidence in the match. The list is sorted by decreasing
    /// confidence.
    ///
    /// * 100% means that the name of the reference entry perfectly matches the
    ///   CRS name, and both are equivalent. In which case a single result is
    ///   returned. Note: in the case of a GeographicCRS whose axis order is
    ///   implicit in the input definition (for example ESRI WKT), then axis
    ///   order is ignored for the purpose of identification. That is the CRS
    ///   built from
    ///   GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.
    ///   0, 298.257223563]],
    ///   PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]] will be
    ///   identified to EPSG:4326, but will not pass a isEquivalentTo(EPSG_4326,
    ///   util::IComparable::Criterion::EQUIVALENT) test, but rather
    ///   isEquivalentTo(EPSG_4326,
    ///   util::IComparable::Criterion::EQUIVALENT_EXCEPT_AXIS_ORDER_GEOGCRS)
    /// * 90% means that CRS are equivalent, but the names are not exactly the
    ///   same.
    /// * 70% means that CRS are equivalent, but the names are not equivalent.
    /// * 25% means that the CRS are not equivalent, but there is some
    ///   similarity in the names.
    ///
    ///Other confidence values may be returned by some specialized
    /// implementations.
    ///
    /// This is implemented for GeodeticCRS, ProjectedCRS,
    /// VerticalCRS and CompoundCRS. Return the hub CRS of a BoundCRS or the
    /// target CRS of a CoordinateOperation.
    ///
    /// # Arguments
    ///
    /// * `auth_name`: Authority name, or NULL for all authorities
    ///
    ///# References
    ///
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_identify>
    pub fn identify(&self, auth_name: &str) -> miette::Result<ProjObjList> {
        let mut confidence: Vec<i32> = Vec::new();
        let result = unsafe {
            proj_sys::proj_identify(
                self.ctx.ptr,
                self.ptr(),
                auth_name.to_cstring().as_ptr(),
                ptr::null(),
                &mut confidence.as_mut_ptr(),
            )
        };
        ProjObjList::new(&self.ctx, result)
    }
}
#[cfg(test)]
mod test_context {
    use super::*;
    #[test]
    fn test_create_from_name() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj_list = ctx.create_from_name(None, "WGS 84", None, false, 0)?;
        println!(
            "{}",
            pj_list
                .get(0)?
                .as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?
        );

        Ok(())
    }
    #[test]
    fn test_query_geodetic_crs_from_datum() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj_list =
            ctx.query_geodetic_crs_from_datum(Some("EPSG"), "EPSG", "6326", Some("geographic 2D"))?;
        println!(
            "{}",
            pj_list
                .get(0)?
                .as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?
        );

        Ok(())
    }
}
#[cfg(test)]
mod test_proj {
    use super::*;
    #[test]
    fn test_get_non_deprecated() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4226")?;
        let pj_list = pj.get_non_deprecated()?;
        println!(
            "{}",
            pj_list
                .get(0)?
                .as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?
        );

        Ok(())
    }
    #[test]
    fn test_identify() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let pj = ctx.create("EPSG:4326")?;
        let pj_list = pj.identify("EPSG")?;
        println!(
            "{}",
            pj_list
                .get(0)?
                .as_wkt(WktType::Wkt2_2019, None, None, None, None, None, None)?
        );

        Ok(())
    }
}
