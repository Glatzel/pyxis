pub enum PJParams<'a> {
    // Transformation setup
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create>
    ByDefination {
        definition: &'a str,
    },
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv>
    ByArgv {
        argv: Vec<&'a str>,
    },
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs>
    ByCrsToCrs {
        source_crs: &'a str,
        target_crs: &'a str,
        area: &'a crate::PjArea,
    },
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs_from_pj>
    ByCrsToCrsFromPj {
        source_crs: crate::Proj,
        target_crs: crate::Proj,
        area: &'a crate::PjArea,
        authority: Option<&'a str>,
        accuracy: Option<f64>,
        allow_ballpark: Option<bool>,
        only_best: Option<bool>,
        force_over: Option<bool>,
    },
    //Iso19111

    // Extension
    ByEpsgCode {
        code: u32,
    },
}
/// Proj Creation
impl crate::PjContext {
    fn create_epsg_code(&self, code: u32) -> miette::Result<crate::Proj> {
        self.create(&format!("EPSG:{code}"))
    }
    pub fn create_proj(&self, by: PJParams) -> miette::Result<crate::Proj> {
        match by {
            // Transformation setup
            PJParams::ByDefination { definition } => self.create(definition),
            PJParams::ByArgv { argv } => self.create_argv(argv.as_slice()),
            PJParams::ByCrsToCrs {
                source_crs,
                target_crs,
                area,
            } => self.create_crs_to_crs(source_crs, target_crs, area),
            PJParams::ByCrsToCrsFromPj {
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
            PJParams::ByEpsgCode { code } => self.create_epsg_code(code),
        }
    }
}
#[cfg(test)]
mod test {
    #[test]
    fn test_create_epsg_code() -> miette::Result<()> {
        let ctx = crate::new_test_ctx();
        ctx.create_epsg_code(4326)?;
        Ok(())
    }
}
