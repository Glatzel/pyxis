impl crate::Proj<'_> {
    pub(crate) fn from_raw(
        ctx: &crate::Context,
        ptr: *mut proj_sys::PJ,
    ) -> miette::Result<crate::Proj<'_>> {
        if ptr.is_null() {
            miette::bail!("Proj pointer is null.");
        }
        Ok(crate::Proj { ctx, ptr })
    }
    pub fn assert_crs(&self) -> miette::Result<&Self> {
        if !self.is_crs() {
            miette::bail!("Proj object is not CRS.");
        }
        Ok(self)
    }
}
