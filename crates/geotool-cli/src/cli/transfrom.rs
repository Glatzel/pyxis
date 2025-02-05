use std::{fmt, path::PathBuf, str::FromStr};

use bpaf::Bpaf;
use miette::IntoDiagnostic;

use std::sync::LazyLock;

#[derive(Debug, Clone, Copy)]
pub enum CoordSpace {
    Cartesian,
    Cylindrical,
    Spherical,
}
impl FromStr for CoordSpace {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cartesian" => Ok(Self::Cartesian),
            "cylindrical" => Ok(Self::Cylindrical),
            "spherical" => Ok(Self::Spherical),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for CoordSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cartesian => write!(f, "Cartesian"),
            Self::Cylindrical => write!(f, "Cylindrical"),
            Self::Spherical => write!(f, "Spherical"),
        }
    }
}
#[derive(Debug, Clone, Copy, Bpaf)]
pub enum CryptoSpace {
    BD09,
    GCJ02,
    WGS84,
}

impl FromStr for CryptoSpace {
    type Err = miette::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_uppercase().as_str() {
            "BD09" => Ok(Self::BD09),
            "GCJ02" => Ok(Self::GCJ02),
            "WGS84" => Ok(Self::WGS84),
            _ => miette::bail!(""),
        }
    }
}
impl fmt::Display for CryptoSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BD09 => write!(f, "BD09"),
            Self::GCJ02 => write!(f, "GCJ02"),
            Self::WGS84 => write!(f, "WGS84"),
        }
    }
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
        println!("    Crypto from: {}", from);
        println!("    Crypto to: {}", to);
        println!(
            "longitude: {}, latitude: {}, elevation:{}",
            self.x, self.y, self.z
        );
    }
    pub fn cvt_proj(&mut self, from: &str, to: &str) -> miette::Result<()> {
        let transformer = Self::init_proj_builder()
            .proj_known_crs(from, to, None)
            .into_diagnostic()?;
        (self.x, self.y) = transformer.convert((self.x, self.y)).into_diagnostic()?;
        println!("    Proj from: {}", from);
        println!("    Proj to: {}", to);
        println!("x: {}, y: {}, z: {}", self.x, self.y, self.z);
        Ok(())
    }
    pub fn datum_compense(&mut self, hb: f64, r: f64, x0: f64, y0: f64) {
        (self.x, self.y) = geotool_algorithm::datum_compense(self.x, self.y, hb, r, x0, y0);
        println!("    datum_compense: hb[{hb}] , r[{r}] , x0[{x0}], y0[{y0}]");
        println!("x: {}, y: {}, elevation: {}", self.x, self.y, self.z);
    }
    pub fn lbh2xyz(&mut self, semi_major_axis: f64, inverse_flattening: f64) {
        (self.x, self.y, self.z) =
            geotool_algorithm::lbh2xyz(self.x, self.y, self.z, semi_major_axis, inverse_flattening);
        println!("    From Geodetic to Geocentric");
        println!(
            "longitude: {}, latitude: {}, elevation: {}",
            self.x, self.y, self.z
        );
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
pub fn execute(x: f64, y: f64, z: f64, cmds: Vec<TransformCommands>) {
    let mut ctx = ContextTransform { x, y, z };
    println!("x: {}, y: {}, z:{}", ctx.x, ctx.y, ctx.z);
    for cmd in cmds {
        match cmd {
            TransformCommands::Crypto { from, to } => ctx.crypto(from, to),
            TransformCommands::DatumCompense {
                hb,
                radius: r,
                x0,
                y0,
            } => ctx.datum_compense(hb, r, x0, y0),
            TransformCommands::Lbh2xyz {
                major_radius: semi_major_axis,
                inverse_flattening,
            } => ctx.lbh2xyz(semi_major_axis, inverse_flattening),
            TransformCommands::Proj { from, to } => {
                ctx.cvt_proj(from.as_str(), to.as_str()).unwrap()
            }
            TransformCommands::Xyz2lbh {
                major_radius: semi_major_axis,
                inverse_flattening,
            } => ctx.xyz2lbh(semi_major_axis, inverse_flattening, None, None),
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
