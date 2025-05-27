use crate::Proj;

pub enum PjParams<'a> {
    // Transformation setup
    ///See [`crate::Context::create`]
    ///
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create>
    Definition(&'a str),
    ///See [`crate::Context::create_argv`]
    ///
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_argv>
    Argv(Vec<&'a str>),
    ///See [`crate::Context::create_crs_to_crs`]
    ///
    ///<https://proj.org/en/stable/development/reference/functions.html#c.proj_create_crs_to_crs>
    CrsToCrs {
        source_crs: &'a str,
        target_crs: &'a str,
        area: &'a crate::Area,
    },
    ///See [`crate::Context::create_crs_to_crs_from_pj`]
    ///
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
    ///See [`crate::Context::create_cs`]
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_cs>
    Cs {
        coordinate_system_type: crate::data_types::iso19111::CoordinateSystemType,
        axis: Vec<crate::data_types::iso19111::AxisDescription>,
    },
    ///See [`crate::Context::create_cartesian_2d_cs`]
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_cartesian_2D_cs>
    Cartesian2dCs {
        ellipsoidal_cs_2d_type: crate::data_types::iso19111::CartesianCs2dType,
        unit_name: Option<&'a str>,
        unit_conv_factor: f64,
    },
    ///See [`crate::Context::create_ellipsoidal_2d_cs`]
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_ellipsoidal_2D_cs>
    Ellipsoidal2dCs {
        ellipsoidal_cs_2d_type: crate::data_types::iso19111::EllipsoidalCs2dType,
        unit_name: Option<&'a str>,
        unit_conv_factor: f64,
    },
    ///See [`crate::Context::create_ellipsoidal_3d_cs`]
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_ellipsoidal_3D_cs>
    Ellipsoidal3dCs {
        ellipsoidal_cs_3d_type: crate::data_types::iso19111::EllipsoidalCs3dType,
        horizontal_angular_unit_name: Option<&'a str>,
        horizontal_angular_unit_conv_factor: f64,
        vertical_linear_unit_name: Option<&'a str>,
        vertical_linear_unit_conv_factor: f64,
    },
    ///See [`crate::Context::create_geographic_crs`]
    ///
    /// <https://proj.org/en/stable/development/reference/functions.html#c.proj_create_ellipsoidal_3D_cs>
    GeographicCrs {
        crs_name: Option<&'a str>,
        datum_name: Option<&'a str>,
        ellps_name: Option<&'a str>,
        semi_major_metre: f64,
        inv_flattening: f64,
        prime_meridian_name: Option<&'a str>,
        prime_meridian_offset: f64,
        pm_angular_units: Option<&'a str>,
        pm_units_conv: f64,
        ellipsoidal_cs: &'a Proj<'a>,
    },
    // Extension
    ///See [`crate::Context::create_epsg_code`]
    EpsgCode(u32),
}

/// Proj Creation
impl crate::Context {
    pub fn create_epsg_code(&self, code: u32) -> miette::Result<crate::Proj> {
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
            } => self.create_cs(coordinate_system_type, &axis),
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
            PjParams::GeographicCrs {
                crs_name,
                datum_name,
                ellps_name,
                semi_major_metre,
                inv_flattening,
                prime_meridian_name,
                prime_meridian_offset,
                pm_angular_units,
                pm_units_conv,
                ellipsoidal_cs,
            } => self.create_geographic_crs(
                crs_name,
                datum_name,
                ellps_name,
                semi_major_metre,
                inv_flattening,
                prime_meridian_name,
                prime_meridian_offset,
                pm_angular_units,
                pm_units_conv,
                ellipsoidal_cs,
            ),

            // Extension
            PjParams::EpsgCode(code) => self.create_epsg_code(code),
        }
    }
}
#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_proj_creation() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        // Transformation setup
        ctx.create_proj(PjParams::Definition("EPSG:4326"))?;
        ctx.create_proj(PjParams::Argv(vec!["proj=utm", "zone=32", "ellps=GRS80"]))?;
        ctx.create_proj(PjParams::CrsToCrs {
            source_crs: "EPSG:4326",
            target_crs: "EPSG:4978",
            area: &crate::Area::default(),
        })?;
        ctx.create_proj(PjParams::CrsToCrsFromPj {
            source_crs: ctx.create("EPSG:4326")?,
            target_crs: ctx.create("EPSG:4978")?,
            area: &crate::Area::default(),
            authority: Some("any"),
            accuracy: Some(0.001),
            allow_ballpark: Some(true),
            only_best: Some(true),
            force_over: Some(true),
        })?;

        // Iso19111
        ctx.create_proj(PjParams::Cs {
            coordinate_system_type: crate::data_types::iso19111::CoordinateSystemType::Cartesian,
            axis: vec![
                crate::data_types::iso19111::AxisDescription::new(
                    Some("Longitude"),
                    Some("lon"),
                    crate::data_types::iso19111::AxisDirection::East,
                    Some("Degree"),
                    1.0,
                    crate::data_types::iso19111::UnitType::Angular,
                ),
                crate::data_types::iso19111::AxisDescription::new(
                    Some("Latitude"),
                    Some("lat"),
                    crate::data_types::iso19111::AxisDirection::North,
                    Some("Degree"),
                    1.0,
                    crate::data_types::iso19111::UnitType::Angular,
                ),
            ],
        })?;
        ctx.create_proj(PjParams::Cartesian2dCs {
            ellipsoidal_cs_2d_type: crate::data_types::iso19111::CartesianCs2dType::EastingNorthing,
            unit_name: Some("Degree"),
            unit_conv_factor: 1.0,
        })?;
        ctx.create_proj(PjParams::Ellipsoidal2dCs {
            ellipsoidal_cs_2d_type:
                crate::data_types::iso19111::EllipsoidalCs2dType::LatitudeLongitude,
            unit_name: Some("Degree"),
            unit_conv_factor: 1.0,
        })?;
        ctx.create_proj(PjParams::Ellipsoidal3dCs {
            ellipsoidal_cs_3d_type:
                crate::data_types::iso19111::EllipsoidalCs3dType::LatitudeLongitudeHeight,
            horizontal_angular_unit_name: Some("Degree"),
            horizontal_angular_unit_conv_factor: 1.0,
            vertical_linear_unit_name: Some("Degree"),
            vertical_linear_unit_conv_factor: 1.0,
        })?;
        ctx.create_proj(PjParams::GeographicCrs {
            crs_name: Some("WGS 84"),
            datum_name: Some("World Geodetic System 1984"),
            ellps_name: Some("WGS84"),
            semi_major_metre: 6378137.0,
            inv_flattening: 298.257223563,
            prime_meridian_name: Some("Greenwich"),
            prime_meridian_offset: 1.0,
            pm_angular_units: Some("Degree"),
            pm_units_conv: 1.0,
            ellipsoidal_cs: &ctx.create_ellipsoidal_2d_cs(
                crate::data_types::iso19111::EllipsoidalCs2dType::LatitudeLongitude,
                Some("Degree"),
                1.0,
            )?,
        })?;

        // Extension
        ctx.create_proj(PjParams::EpsgCode(4326))?;
        Ok(())
    }
}
