impl crate::Proj<'_> {
    /// Create a `Proj` object from pointer, panic if pointer is null.
    pub(crate) fn from_raw(
        ctx: &crate::Context,
        ptr: *mut proj_sys::PJ,
    ) -> miette::Result<crate::Proj<'_>> {
        if ptr.is_null() {
            miette::bail!("Proj pointer is null.");
        }
        Ok(crate::Proj { ctx, ptr })
    }
    /// Panic if a `Proj` object is not CRS.
    pub fn assert_crs(&self) -> miette::Result<&Self> {
        if !self.is_crs() {
            miette::bail!("Proj object is not CRS.");
        }
        Ok(self)
    }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_assert_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        //is crs
        {
            let pj = ctx.create("EPSG:4326")?;
            assert!(pj.assert_crs().is_ok());
        }
        //not crs
        {
            let pj = ctx.create("+proj=utm +zone=32 +datum=WGS84")?;
            assert!(pj.assert_crs().is_err());
        }
        Ok(())
    }
}
