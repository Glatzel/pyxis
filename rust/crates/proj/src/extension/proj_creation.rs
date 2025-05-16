pub enum PjParams<'a> {
    // Transformation setup
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create>
    Definition(&'a str),
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv>
    Argv(Vec<&'a str>),
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs>
    CrsToCrs {
        source_crs: &'a str,
        target_crs: &'a str,
        area: &'a crate::Area,
    },
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs_from_pj>
    CrsToCrsFromPj {
        source_crs: crate::Proj<'a>,
        target_crs: crate::Proj<'a>,
        area: &'a crate::Area,
        authority: Option<&'a str>,
        accuracy: Option<f64>,
        allow_ballpark: Option<bool>,
        only_best: Option<bool>,
        force_over: Option<bool>,
    },
    //Iso19111

    // Extension
    EpsgCode(u32),
}

/// Proj Creation
impl crate::Context {
    fn create_epsg_code(&self, code: u32) -> miette::Result<crate::Proj> {
        self.create(&format!("EPSG:{code}"))
    }
    pub fn create_proj(&self, by: PjParams) -> miette::Result<crate::Proj> {
        match by {
            // Transformation setup
            PjParams::Definition(definition) => self.create(definition),
            PjParams::Argv(argv) => self.create_argv(argv.as_slice()),
            PjParams::CrsToCrs {
                source_crs,
                target_crs,
                area,
            } => self.create_crs_to_crs(source_crs, target_crs, area),
            PjParams::CrsToCrsFromPj {
                source_crs,
                target_crs,
                area,
                authority,
                accuracy,
                allow_ballpark,
                only_best,
                force_over,
            } => self.create_crs_to_crs_from_pj(
                source_crs,
                target_crs,
                area,
                authority,
                accuracy,
                allow_ballpark,
                only_best,
                force_over,
            ),
            // Iso19111
            // Extension
            PjParams::EpsgCode(code) => self.create_epsg_code(code),
        }
    }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_create_epsg_code() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        ctx.create_epsg_code(4326)?;
        Ok(())
    }
}
