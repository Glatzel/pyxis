mod options;
use bpaf::Bpaf;
use miette::IntoDiagnostic;
pub use options::*;
use std::{path::PathBuf, sync::LazyLock};
pub struct Record {
    pub idx: u8,
    pub method: String,
    pub from: String,
    pub to: String,

    pub ox: f64,
    pub oy: f64,
    pub oz: f64,
    pub ox_name: String,
    pub oy_name: String,
    pub oz_name: String,
}

pub struct ContextTransform {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
static PROJ_RESOURCE_PATH: LazyLock<PathBuf> = LazyLock::new(|| {
    let exe_path = std::env::current_exe().unwrap();
    PathBuf::from(exe_path.parent().unwrap())
});
impl ContextTransform {
    fn init_proj_builder() -> proj::ProjBuilder {
        let mut builder = proj::ProjBuilder::new();
        builder
            .set_search_paths(PROJ_RESOURCE_PATH.as_path())
            .unwrap();
        builder
    }
    pub fn _to_space(&mut self, from: CoordSpace, to: CoordSpace) {
        (self.x, self.y, self.z) = match (from, to) {
            (CoordSpace::Cartesian, CoordSpace::Cylindrical) => {
                geotool_algorithm::cartesian_to_cylindrical(self.x, self.y, self.z)
            }
            (CoordSpace::Cartesian, CoordSpace::Spherical) => {
                geotool_algorithm::cartesian_to_spherical(self.x, self.y, self.z)
            }
            (CoordSpace::Cylindrical, CoordSpace::Cartesian) => {
                geotool_algorithm::cylindrical_to_cartesian(self.x, self.y, self.z)
            }
            (CoordSpace::Cylindrical, CoordSpace::Spherical) => {
                geotool_algorithm::cylindrical_to_spherical(self.x, self.y, self.z)
            }
            (CoordSpace::Spherical, CoordSpace::Cartesian) => {
                geotool_algorithm::spherical_to_cartesian(self.x, self.y, self.z)
            }
            (CoordSpace::Spherical, CoordSpace::Cylindrical) => {
                geotool_algorithm::spherical_to_cylindrical(self.x, self.y, self.z)
            }
            _ => {
                tracing::warn!("Nothing changes from <{from}> to <{to}>.");
                (self.x, self.y, self.z)
            }
        };
    }
    pub fn crypto(&mut self, from: CryptoSpace, to: CryptoSpace) {
        (self.x, self.y) = match (from, to) {
            (CryptoSpace::BD09, CryptoSpace::GCJ02) => {
                geotool_algorithm::bd09_to_gcj02(self.x, self.y)
            }
            (CryptoSpace::BD09, CryptoSpace::WGS84) => {
                geotool_algorithm::bd09_to_wgs84(self.x, self.y)
            }
            (CryptoSpace::GCJ02, CryptoSpace::BD09) => {
                geotool_algorithm::gcj02_to_bd09(self.x, self.y)
            }
            (CryptoSpace::GCJ02, CryptoSpace::WGS84) => {
                geotool_algorithm::gcj02_to_wgs84(self.x, self.y)
            }
            (CryptoSpace::WGS84, CryptoSpace::BD09) => {
                geotool_algorithm::wgs84_to_bd09(self.x, self.y)
            }
            (CryptoSpace::WGS84, CryptoSpace::GCJ02) => {
                geotool_algorithm::wgs84_to_gcj02(self.x, self.y)
            }
            _ => {
                tracing::warn!("Nothing changes from <{from}> to <{to}>.");
                (self.x, self.y)
            }
        };
    }
    pub fn cvt_proj(&mut self, from: &str, to: &str) -> miette::Result<()> {
        let transformer = Self::init_proj_builder()
            .proj_known_crs(from, to, None)
            .into_diagnostic()?;
        (self.x, self.y) = transformer.convert((self.x, self.y)).into_diagnostic()?;
        Ok(())
    }
    pub fn datum_compense(&mut self, hb: f64, r: f64, x0: f64, y0: f64) {
        (self.x, self.y) = geotool_algorithm::datum_compense(self.x, self.y, hb, r, x0, y0);
    }
    pub fn lbh2xyz(&mut self, semi_major_axis: f64, inverse_flattening: f64) {
        (self.x, self.y, self.z) =
            geotool_algorithm::lbh2xyz(self.x, self.y, self.z, semi_major_axis, inverse_flattening);
    }
    pub fn xyz2lbh(
        &mut self,
        semi_major_axis: f64,
        inverse_flattening: f64,
        tolerance: Option<f64>,
        max_iterations: Option<u32>,
    ) {
        (self.x, self.y, self.z) = geotool_algorithm::xyz2lbh(
            self.x,
            self.y,
            self.z,
            semi_major_axis,
            inverse_flattening,
            tolerance,
            max_iterations,
        );
    }
}

#[derive(Bpaf, Clone, Debug)]
pub enum TransformCommands {
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Crypto coordinates between `BD09`, `GCJ02` and `WGS84`.
    Crypto {
        #[bpaf(short, long)]
        from: CryptoSpace,
        #[bpaf(short, long)]
        to: CryptoSpace,
    },
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Converts projected XY coordinates from the height compensation plane to the sea level plane.
    DatumCompense {
        #[bpaf(long)]
        /// Elevation of the height compensation plane (in meters).
        hb: f64,
        #[bpaf(short, long)]
        /// Radius of the Earth (in meters).
        radius: f64,
        #[bpaf(long)]
        /// X coordinate system origin (in meters).
        x0: f64,
        #[bpaf(long)]
        /// Y coordinate system origin (in meters).
        y0: f64,
    },
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Converts geodetic coordinates (longitude/L, latitude/B, height/H) to Cartesian coordinates (X, Y, Z).
    Lbh2xyz {
        #[bpaf(short('a'), long)]
        /// Semimajor radius of the ellipsoid axis
        major_radius: f64,
        #[bpaf(long("invf"))]
        /// Inverse flattening of the ellipsoid.
        inverse_flattening: f64,
    },
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Transform coordinate from one known coordinate reference systems to another.
    ///
    ///  The `from` and `to` can be:
    ///  - an "AUTHORITY:CODE", like "EPSG:25832".
    ///  - a PROJ string, like "+proj=longlat +datum=WGS84". When using that syntax, the unit is expected to be degrees.
    ///  - the name of a CRS as found in the PROJ database, e.g "WGS84", "NAD27", etc.
    Proj {
        #[bpaf(short, long, argument("PROJ"))]
        from: String,
        #[bpaf(short, long, argument("PROJ"))]
        to: String,
    },
    #[bpaf(command, adjacent, fallback_to_usage)]
    /// Converts Cartesian coordinates (X, Y, Z) to geodetic coordinates (Longitude, Latitude, Height).
    Xyz2lbh {
        #[bpaf(short('a'), long)]
        /// Semimajor radius of the ellipsoid axis
        major_radius: f64,
        #[bpaf(long("invf"))]
        /// Inverse flattening of the ellipsoid.
        inverse_flattening: f64,
    },
}
pub fn execute(x: f64, y: f64, z: f64, output_format: OutputFormat, cmds: Vec<TransformCommands>) {
    let mut ctx = ContextTransform { x, y, z };
    let mut records: Vec<Record> = vec![Record {
        idx: 0,
        method: "input".to_string(),
        from: "".to_string(),
        to: "".to_string(),
        ox: ctx.x,
        oy: ctx.y,
        oz: ctx.z,
        ox_name: "x".to_string(),
        oy_name: "y".to_string(),
        oz_name: "z".to_string(),
    }];
    println!("x: {}, y: {}, z:{}", ctx.x, ctx.y, ctx.z);
    for (i, cmd) in cmds.iter().enumerate() {
        match cmd {
            TransformCommands::Crypto { from, to } => {
                ctx.crypto(*from, *to);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "crypto".to_string(),
                    from: from.to_string(),
                    to: to.to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "longitude".to_string(),
                    oy_name: "latitude".to_string(),
                    oz_name: "elevation".to_string(),
                };
                records.push(record);
            }
            TransformCommands::DatumCompense {
                hb,
                radius: r,
                x0,
                y0,
            } => {
                ctx.datum_compense(*hb, *r, *x0, *y0);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "datum compense".to_string(),
                    from: "".to_string(),
                    to: "".to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "x".to_string(),
                    oy_name: "y".to_string(),
                    oz_name: "elevation".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Lbh2xyz {
                major_radius: semi_major_axis,
                inverse_flattening,
            } => {
                ctx.lbh2xyz(*semi_major_axis, *inverse_flattening);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "lbh2xyz".to_string(),
                    from: "lbh".to_string(),
                    to: "xyz".to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "x".to_string(),
                    oy_name: "y".to_string(),
                    oz_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Proj { from, to } => {
                ctx.cvt_proj(from.as_str(), to.as_str()).unwrap();
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "proj".to_string(),
                    from: from.to_string(),
                    to: to.to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "x".to_string(),
                    oy_name: "y".to_string(),
                    oz_name: "z".to_string(),
                };
                records.push(record);
            }
            TransformCommands::Xyz2lbh {
                major_radius: semi_major_axis,
                inverse_flattening,
            } => {
                ctx.xyz2lbh(*semi_major_axis, *inverse_flattening, None, None);
                let record = Record {
                    idx: (i + 1) as u8,
                    method: "xyz2lbh".to_string(),
                    from: "xyz".to_string(),
                    to: "lbh".to_string(),
                    ox: ctx.x,
                    oy: ctx.y,
                    oz: ctx.z,
                    ox_name: "longitude".to_string(),
                    oy_name: "latitude".to_string(),
                    oz_name: "elevation".to_string(),
                };
                records.push(record);
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use float_cmp::approx_eq;

    use super::*;

    #[test]
    fn test_zygs() {
        let mut ctx = ContextTransform {
            x: 469704.6693,
            y: 2821940.796,
            z: 0.0,
        };
        ctx.datum_compense(400.0, 6_378_137.0, 500_000.0, 0.0);
        ctx.cvt_proj("+proj=tmerc +lat_0=0 +lon_0=118.5 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs", "+proj=longlat +datum=WGS84 +no_defs +type=crs").unwrap();
        println!("x:{}, y:{}, z:{}", ctx.x, ctx.y, ctx.z);
        assert!(approx_eq!(f64, ctx.x, 118.19868034481004, epsilon = 1e-6));
        assert!(approx_eq!(f64, ctx.y, 25.502591181714727, epsilon = 1e-6));
        assert!(approx_eq!(f64, ctx.z, 0.0, epsilon = 1e-6));
    }
    #[test]
    fn test_jxws() {
        let mut ctx = ContextTransform {
            x: 121.091701,
            y: 30.610765,
            z: 0.0,
        };
        ctx.crypto(CryptoSpace::WGS84, CryptoSpace::GCJ02);

        println!("x:{}, y:{}, z:{}", ctx.x, ctx.y, ctx.z);
        assert!(approx_eq!(f64, ctx.x, 121.09626257405186, epsilon = 1e-6));
        assert!(approx_eq!(f64, ctx.y, 30.608591461324128, epsilon = 1e-6));
        assert!(approx_eq!(f64, ctx.z, 0.0, epsilon = 1e-6));
    }
}
