/// # Logging
impl crate::Context {
    /// # See Also
    ///
    /// * [`Self::set_log_level`]
    ///
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_log_level>
    fn _log_level(&self) { unimplemented!("Use other function to instead.") }
    /// # See Also
    ///
    /// * [`Self::set_log_fn`]
    ///
    /// # References
    /// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_log_func>
    fn _log_func(&self) -> miette::Result<&Self> {
        unimplemented!("Use other function to instead.")
    }
}
