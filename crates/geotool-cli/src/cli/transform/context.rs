use std::path::PathBuf;
use std::sync::LazyLock;

use geotool_algorithm::Ellipsoid;
use miette::IntoDiagnostic;

use super::{options, CoordSpace, CryptoSpace, MigrateOption2d, RotateUnit};
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
    pub fn migrate2d(
        &mut self,
        given: MigrateOption2d,
        another: MigrateOption2d,
        another_x: f64,
        another_y: f64,
        rotate: f64,
        unit: RotateUnit,
    ) {
        let rotate = match unit {
            options::RotateUnit::Angle => rotate.to_radians(),
            _ => rotate,
        };

        let rotate_matrix = geotool_algorithm::rotate_matrix_2d(rotate);
        (self.x, self.y) = match (given, another) {
            (MigrateOption2d::Absolute, MigrateOption2d::Origin) => {
                geotool_algorithm::migrate::rel_2d(
                    another_x,
                    another_y,
                    self.x,
                    self.y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Origin, MigrateOption2d::Absolute) => {
                geotool_algorithm::migrate::rel_2d(
                    self.x,
                    self.y,
                    another_x,
                    another_y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Absolute, MigrateOption2d::Relative) => {
                geotool_algorithm::migrate::origin_2d(
                    self.x,
                    self.y,
                    another_x,
                    another_y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Relative, MigrateOption2d::Absolute) => {
                geotool_algorithm::migrate::origin_2d(
                    another_x,
                    another_y,
                    self.x,
                    self.y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Relative, MigrateOption2d::Origin) => {
                geotool_algorithm::migrate::abs_2d(
                    another_x,
                    another_y,
                    self.x,
                    self.y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Origin, MigrateOption2d::Relative) => {
                geotool_algorithm::migrate::abs_2d(
                    self.x,
                    self.y,
                    another_x,
                    another_y,
                    &rotate_matrix,
                )
            }

            (given, another) => {
                tracing::warn!("Given and anther is the same. Given: {given}, Another: {another}.");
                (self.x, self.y)
            }
        }
    }
    pub fn normalize(&mut self) {
        if self.x == 0.0f64 && self.y == 0.0f64 && self.z == 0.0f64 {
            tracing::warn!("Length of coordinate vector is 0.")
        } else {
            let length = (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt();
            self.x /= length;
            self.y /= length;
            self.z /= length;
        }
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
    pub fn scale(&mut self, sx: f64, sy: f64, sz: f64, ox: f64, oy: f64, oz: f64) {
        if ox == 0.0f64 && oy == 0.0f64 && oz == 0.0f64 {
            tracing::warn!("Translation parameters are all 0. The Coordinate is not modified after translation.")
        }
        self.x = ox + (self.x - ox) * sx;
        self.y = oy + (self.y - oy) * sy;
        self.z = oz + (self.z - oz) * sz;
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
    pub fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        if tx == 0.0f64 && ty == 0.0f64 && tz == 0.0f64 {
            tracing::warn!("Translation parameters are all 0. The Coordinate is not modified after translation.")
        }
        self.x += tx;
        self.y += ty;
        self.z += tz;
    }
    pub fn xyz2lbh(&mut self, ellipsoid: &Ellipsoid) {
        (self.x, self.y, self.z) = geotool_algorithm::xyz2lbh(self.x, self.y, self.z, ellipsoid);
    }
}
