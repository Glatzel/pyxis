use std::path::PathBuf;
use std::sync::LazyLock;

use geotool_algorithm::Ellipsoid;
use miette::IntoDiagnostic;

use super::{options, CoordSpace, CryptoSpace};
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
    pub fn crypto(&mut self, from: CryptoSpace, to: CryptoSpace) {
        (self.x, self.y) = match (from, to) {
            (CryptoSpace::BD09, CryptoSpace::GCJ02) => {
                geotool_algorithm::bd09_to_gcj02_exact(self.x, self.y, 1e-17, 1000)
            }
            (CryptoSpace::BD09, CryptoSpace::WGS84) => {
                geotool_algorithm::bd09_to_wgs84_exact(self.x, self.y, 1e-17, 1000)
            }
            (CryptoSpace::GCJ02, CryptoSpace::BD09) => {
                geotool_algorithm::gcj02_to_bd09(self.x, self.y)
            }
            (CryptoSpace::GCJ02, CryptoSpace::WGS84) => {
                geotool_algorithm::gcj02_to_wgs84_exact(self.x, self.y, 1e-17, 1000)
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

    pub fn datum_compense(&mut self, hb: f64, r: f64, x0: f64, y0: f64) {
        (self.x, self.y) = geotool_algorithm::datum_compense(self.x, self.y, hb, r, x0, y0);
    }
    pub fn lbh2xyz(&mut self, ellipsoid: &Ellipsoid) {
        (self.x, self.y, self.z) = geotool_algorithm::lbh2xyz(self.x, self.y, self.z, ellipsoid);
    }
    pub fn normalize(&mut self) {
        let length = (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt();
        self.x /= length;
        self.y /= length;
        self.z /= length;
    }
    pub fn proj(&mut self, from: &str, to: &str) -> miette::Result<()> {
        let transformer = Self::init_proj_builder()
            .proj_known_crs(from, to, None)
            .into_diagnostic()?;
        (self.x, self.y) = transformer.convert((self.x, self.y)).into_diagnostic()?;
        Ok(())
    }
    pub fn rotate(&mut self, r: f64, axis: options::RotateAxis, unit: options::RotateUnit) {
        let r = match unit {
            options::RotateUnit::Angle => r.to_radians(),
            _ => r,
        };
        let m = geotool_algorithm::rotate_matrix_2d(r);
        match axis {
            super::RotateAxis::Xy => {
                (self.x, self.y) = geotool_algorithm::rotate_2d(self.x, self.y, &m);
            }
            super::RotateAxis::Zx => {
                (self.z, self.x) = geotool_algorithm::rotate_2d(self.z, self.x, &m);
            }
            super::RotateAxis::Yz => {
                (self.y, self.z) = geotool_algorithm::rotate_2d(self.x, self.y, &m);
            }
        }
    }
    pub fn scale(&mut self, x: f64, y: f64, z: f64) {
        self.x *= x;
        self.y *= y;
        self.z *= z;
    }
    pub fn space(&mut self, from: CoordSpace, to: CoordSpace) {
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
    pub fn translate(&mut self, x: f64, y: f64, z: f64) {
        self.x += x;
        self.y += y;
        self.z += z;
    }
    pub fn xyz2lbh(&mut self, ellipsoid: &Ellipsoid) {
        (self.x, self.y, self.z) = geotool_algorithm::xyz2lbh(self.x, self.y, self.z, ellipsoid);
    }
}
#[cfg(test)]
mod tests {
    use float_cmp::assert_approx_eq;

    use super::*;

    #[test]
    fn test_zygs() {
        let mut ctx = ContextTransform {
            x: 469704.6693,
            y: 2821940.796,
            z: 0.0,
        };
        ctx.datum_compense(400.0, 6_378_137.0, 500_000.0, 0.0);
        ctx.proj("+proj=tmerc +lat_0=0 +lon_0=118.5 +k=1 +x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs +type=crs", "+proj=longlat +datum=WGS84 +no_defs +type=crs").unwrap();
        println!("x:{}, y:{}, z:{}", ctx.x, ctx.y, ctx.z);
        assert_approx_eq!(f64, ctx.x, 118.19868034481004, epsilon = 1e-6);
        assert_approx_eq!(f64, ctx.y, 25.502591181714727, epsilon = 1e-6);
        assert_approx_eq!(f64, ctx.z, 0.0, epsilon = 1e-6);
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
        assert_approx_eq!(f64, ctx.x, 121.09626257405186, epsilon = 1e-6);
        assert_approx_eq!(f64, ctx.y, 30.608591461324128, epsilon = 1e-6);
        assert_approx_eq!(f64, ctx.z, 0.0, epsilon = 1e-6);
    }
}
