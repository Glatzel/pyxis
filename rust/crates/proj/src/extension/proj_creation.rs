use crate::data_types::iso19111::UnitName;

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
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_cs>
    Cs {
        coordinate_system_type: crate::data_types::iso19111::CoordinateSystemType,
        axis: &'a [crate::data_types::iso19111::AxisDescription],
    },
    Cartesian2dCs {
        ellipsoidal_cs_2d_type: crate::data_types::iso19111::CartesianCs2dType,
        unit_name: UnitName,
        unit_conv_factor: f64,
    },
    Ellipsoidal2dCs {
        ellipsoidal_cs_2d_type: crate::data_types::iso19111::EllipsoidalCs2dType,
        unit_name: UnitName,
        unit_conv_factor: f64,
    },
    Ellipsoidal3dCs {
        ellipsoidal_cs_3d_type: crate::data_types::iso19111::EllipsoidalCs3dType,
        horizontal_angular_unit_name: UnitName,
        horizontal_angular_unit_conv_factor: f64,
        vertical_linear_unit_name: UnitName,
        vertical_linear_unit_conv_factor: f64,
    },
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
            PjParams::Cs {
                coordinate_system_type,
                axis,
            } => self.create_cs(coordinate_system_type, axis),
            PjParams::Cartesian2dCs {
                ellipsoidal_cs_2d_type,
                unit_name,
                unit_conv_factor,
            } => self.create_cartesian_2d_cs(ellipsoidal_cs_2d_type, unit_name, unit_conv_factor),
            PjParams::Ellipsoidal2dCs {
                ellipsoidal_cs_2d_type,
                unit_name,
                unit_conv_factor,
            } => self.create_ellipsoidal_2d_cs(ellipsoidal_cs_2d_type, unit_name, unit_conv_factor),
            PjParams::Ellipsoidal3dCs {
                ellipsoidal_cs_3d_type,
                horizontal_angular_unit_name,
                horizontal_angular_unit_conv_factor,
                vertical_linear_unit_name,
                vertical_linear_unit_conv_factor,
            } => self.create_ellipsoidal_3d_cs(
                ellipsoidal_cs_3d_type,
                horizontal_angular_unit_name,
                horizontal_angular_unit_conv_factor,
                vertical_linear_unit_name,
                vertical_linear_unit_conv_factor,
            ),
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
