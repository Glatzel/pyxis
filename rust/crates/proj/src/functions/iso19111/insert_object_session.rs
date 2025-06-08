use std::ptr;

use envoy::{AsVecPtr, CStrListToVecString, ToCString, VecCString};

use crate::data_types::iso19111::InsertObjectSession;
use crate::{Context, Proj};

/// insert object session
impl Context {
    ///Starts a session for
    /// [`InsertObjectSession::get_insert_statements()`].
    ///
    ///
    ///Starts a new session for one or several calls to
    /// [`InsertObjectSession::get_insert_statements()`].
    ///
    ///An insertion session guarantees that the inserted objects will not
    /// create conflicting intermediate objects.
    ///
    /// # References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_insert_object_session_create>
    pub fn insert_object_session_create(&self) -> InsertObjectSession<'_> {
        InsertObjectSession {
            ctx: self,
            ptr: unsafe { proj_sys::proj_insert_object_session_create(self.ptr) },
        }
    }
}
impl Drop for InsertObjectSession<'_> {
    ///Stops an insertion session started with
    /// [`Context::insert_object_session_create()`].
    ///
    /// # References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_insert_object_session_destroy>
    fn drop(&mut self) {
        unsafe { proj_sys::proj_insert_object_session_destroy(self.ctx.ptr, self.ptr) };
    }
}
impl InsertObjectSession<'_> {
    /// # See Also
    ///
    /// * [crate::Context::insert_object_session_create]
    pub fn from_context(ctx: &Context) -> InsertObjectSession<'_> {
        ctx.insert_object_session_create()
    }
    ///Returns SQL statements needed to insert the passed object into the
    /// database.
    ///
    ///It is strongly recommended that new objects should not be added in
    /// common registries, such as "EPSG", "ESRI", "IAU", etc.  Users should
    /// use a custom authority name instead.
    /// If a new object should be added to the official EPSG registry,
    /// users are invited to follow the procedure explained at <https://epsg.org/dataset-change-requests.html>.
    ///
    /// Combined with [`crate::Context::get_database_structure()`],
    /// users can create auxiliary databases, instead of directly modifying
    /// the main proj.db database. Those auxiliary databases can be
    /// specified through [`crate::Context::set_database_path()`] or the
    /// `PROJ_AUX_DB` environment variable.
    ///
    /// # Arguments
    ///
    /// * `object`: The object to insert into the database. Currently only
    ///   PrimeMeridian, Ellipsoid, Datum, GeodeticCRS, ProjectedCRS,
    ///   VerticalCRS, CompoundCRS or BoundCRS are supported.
    /// * `authority`: Authority name into which the object will be inserted. Must
    ///   not be NULL.
    /// * `code`: Code with which the object will be inserted.Must not be NULL.
    /// * `numeric_codes`: Whether intermediate objects that can be created should
    ///   use numeric codes (true), or may be alphanumeric (false)
    /// * `allowed_authorities`: NULL terminated list of authority names, or NULL.
    ///   Authorities to which intermediate objects are allowed to refer to.
    ///   "authority" will be implicitly added to it. Note that unit, coordinate
    ///   systems, projection methods and parameters will in any case be allowed
    ///   to refer to EPSG. If NULL, allowed_authorities defaults to {"EPSG",
    ///   "PROJ", nullptr}
    ///
    /// # Returns
    ///
    /// a list of insert statements.
    ///
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_get_insert_statements>
    pub fn get_insert_statements(
        &self,
        object: &Proj,
        authority: &str,
        code: &str,
        numeric_codes: bool,
        allowed_authorities: Option<&[&str]>,
    ) -> miette::Result<Vec<String>> {
        let allowed_authorities: VecCString = allowed_authorities.into();
        let ptr = unsafe {
            proj_sys::proj_get_insert_statements(
                self.ctx.ptr,
                self.ptr,
                object.ptr(),
                authority.to_cstring().as_ptr(),
                code.to_cstring().as_ptr(),
                numeric_codes as i32,
                allowed_authorities.as_vec_ptr().as_ptr(),
                ptr::null(),
            )
        };
        let result = ptr.to_vec_string();
        unsafe {
            proj_sys::proj_string_list_destroy(ptr);
        }
        Ok(result)
    }
}
#[cfg(test)]
mod test {
    use crate::data_types::iso19111::InsertObjectSession;

    #[test]
    fn test_get_insert_statements() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        let wkt = "GEOGCRS[\"myGDA2020\",
                       DATUM[\"GDA2020\",
                           ELLIPSOID[\"GRS_1980\",6378137,298.257222101,
                               LENGTHUNIT[\"metre\",1]]],
                       PRIMEM[\"Greenwich\",0,
                           ANGLEUNIT[\"Degree\",0.0174532925199433]],
                       CS[ellipsoidal,2],
                           AXIS[\"geodetic latitude (Lat)\",north,
                               ORDER[1],
                               ANGLEUNIT[\"degree\",0.0174532925199433]],
                           AXIS[\"geodetic longitude (Lon)\",east,
                               ORDER[2],
                               ANGLEUNIT[\"degree\",0.0174532925199433]]]";
        println!("{wkt}");
        let crs = ctx.create_from_wkt(wkt, None, None)?;
        let session = InsertObjectSession::from_context(&ctx);
        let statements = session.get_insert_statements(&crs, "HOBU", "XXXX", false, None)?;
        for i in statements.iter() {
            println!("{i}");
        }
        Ok(())
    }
}
