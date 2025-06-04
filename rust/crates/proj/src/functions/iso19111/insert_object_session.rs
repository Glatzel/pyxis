use std::ptr;

use envoy::{CStrListToVecString, ToCStr, ToCStrList};

use crate::data_types::iso19111::InsertObjectSession;
use crate::{Context, Proj};

/// insert object session
impl Context {
    ///# References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_insert_object_session_create>
    pub fn insert_object_session_create(&self) -> InsertObjectSession {
        InsertObjectSession {
            ctx: self,
            ptr: unsafe { proj_sys::proj_insert_object_session_create(self.ptr) },
        }
    }
}
impl Drop for InsertObjectSession<'_> {
    /// # References
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_insert_object_session_destroy>
    fn drop(&mut self) {
        unsafe { proj_sys::proj_insert_object_session_destroy(self.ctx.ptr, self.ptr) };
    }
}
impl InsertObjectSession<'_> {
    pub fn from_context(ctx: &Context) -> InsertObjectSession<'_> {
        ctx.insert_object_session_create()
    }
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
        let allowed_authorities = allowed_authorities.map(|f| f.to_cstr_list());

        let ptr = unsafe {
            proj_sys::proj_get_insert_statements(
                self.ctx.ptr,
                self.ptr,
                object.ptr(),
                authority.to_cstr(),
                code.to_cstr(),
                numeric_codes as i32,
                allowed_authorities.map_or(ptr::null(), |f| f.as_ptr()),
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
