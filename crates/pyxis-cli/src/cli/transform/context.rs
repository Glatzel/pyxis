use pyxis_algorithm::Ellipsoid;
use miette::IntoDiagnostic;

use super::{CoordSpace, CryptoSpace, MigrateOption2d, RotateUnit, options};
pub struct ContextTransform {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ContextTransform {
    pub fn crypto(&mut self, from: CryptoSpace, to: CryptoSpace) {
        (self.x, self.y) = match (from, to) {
            (CryptoSpace::BD09, CryptoSpace::GCJ02) => pyxis_algorithm::crypto::crypto_exact(
                self.x,
                self.y,
                &pyxis_algorithm::crypto::bd09_to_gcj02,
                &pyxis_algorithm::crypto::gcj02_to_bd09,
                1e-17,
                pyxis_algorithm::crypto::CryptoThresholdMode::LonLat,
                1000,
            ),
            (CryptoSpace::BD09, CryptoSpace::WGS84) => pyxis_algorithm::crypto::crypto_exact(
                self.x,
                self.y,
                &pyxis_algorithm::crypto::bd09_to_wgs84,
                &pyxis_algorithm::crypto::wgs84_to_bd09,
                1e-17,
                pyxis_algorithm::crypto::CryptoThresholdMode::LonLat,
                1000,
            ),
            (CryptoSpace::GCJ02, CryptoSpace::BD09) => {
                pyxis_algorithm::crypto::gcj02_to_bd09(self.x, self.y)
            }
            (CryptoSpace::GCJ02, CryptoSpace::WGS84) => pyxis_algorithm::crypto::crypto_exact(
                self.x,
                self.y,
                &pyxis_algorithm::crypto::gcj02_to_wgs84,
                &pyxis_algorithm::crypto::wgs84_to_gcj02,
                1e-17,
                pyxis_algorithm::crypto::CryptoThresholdMode::LonLat,
                1000,
            ),
            (CryptoSpace::WGS84, CryptoSpace::BD09) => {
                pyxis_algorithm::crypto::wgs84_to_bd09(self.x, self.y)
            }
            (CryptoSpace::WGS84, CryptoSpace::GCJ02) => {
                pyxis_algorithm::crypto::wgs84_to_gcj02(self.x, self.y)
            }
            _ => {
                tracing::warn!("Nothing changes from <{from}> to <{to}>.");
                (self.x, self.y)
            }
        };
    }

    pub fn datum_compense(&mut self, hb: f64, r: f64, x0: f64, y0: f64) {
        (self.x, self.y) = pyxis_algorithm::datum_compense(
            self.x,
            self.y,
            &pyxis_algorithm::DatumCompenseParms::new(hb, r, x0, y0),
        );
    }
    pub fn lbh2xyz(&mut self, ellipsoid: &Ellipsoid<f64>) {
        (self.x, self.y, self.z) = pyxis_algorithm::lbh2xyz(self.x, self.y, self.z, ellipsoid);
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
            options::RotateUnit::Degrees => rotate.to_radians(),
            _ => rotate,
        };

        let rotate_matrix = pyxis_algorithm::rotate_matrix_2d(rotate);
        (self.x, self.y) = match (given, another) {
            (MigrateOption2d::Absolute, MigrateOption2d::Origin) => {
                pyxis_algorithm::migrate::rel_2d(
                    another_x,
                    another_y,
                    self.x,
                    self.y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Origin, MigrateOption2d::Absolute) => {
                pyxis_algorithm::migrate::rel_2d(
                    self.x,
                    self.y,
                    another_x,
                    another_y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Absolute, MigrateOption2d::Relative) => {
                pyxis_algorithm::migrate::origin_2d(
                    self.x,
                    self.y,
                    another_x,
                    another_y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Relative, MigrateOption2d::Absolute) => {
                pyxis_algorithm::migrate::origin_2d(
                    another_x,
                    another_y,
                    self.x,
                    self.y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Relative, MigrateOption2d::Origin) => {
                pyxis_algorithm::migrate::abs_2d(
                    another_x,
                    another_y,
                    self.x,
                    self.y,
                    &rotate_matrix,
                )
            }
            (MigrateOption2d::Origin, MigrateOption2d::Relative) => {
                pyxis_algorithm::migrate::abs_2d(
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
        let transformer = crate::proj_util::init_proj_builder()
            .proj_known_crs(from, to, None)
            .into_diagnostic()?;
        (self.x, self.y) = transformer.convert((self.x, self.y)).into_diagnostic()?;
        Ok(())
    }
    pub fn rotate(
        &mut self,
        r: f64,
        plane: options::RotatePlane,
        unit: options::RotateUnit,
        ox: f64,
        oy: f64,
        oz: f64,
    ) {
        let m = match unit {
            options::RotateUnit::Degrees => pyxis_algorithm::rotate_matrix_2d(r.to_radians()),
            _ => pyxis_algorithm::rotate_matrix_2d(r),
        };

        match (plane, unit) {
            (_, RotateUnit::Degrees) if r % 360.0 == 0.0 => tracing::warn!(
                "Rotate angle mod 360 equals 0. The Coordinate is not modified after rotate."
            ),
            (_, RotateUnit::Radians) if r % 360.0 == 0.0 => tracing::warn!(
                "Rotate radians mod PI equals 0. The Coordinate is not modified after rotate."
            ),
            (super::RotatePlane::Xy, _) if ox == self.x && oy == self.y => tracing::warn!(
                "Rotate origin equals to coordinate. The Coordinate is not modified after rotate."
            ),
            (super::RotatePlane::Zx, _) if ox == self.x && oz == self.z => tracing::warn!(
                "Rotate origin equals to coordinate. The Coordinate is not modified after rotate."
            ),
            (super::RotatePlane::Yz, _) if oy == self.y && oz == self.z => tracing::warn!(
                "Rotate origin equals to coordinate. The Coordinate is not modified after rotate."
            ),
            (super::RotatePlane::Xy, _) => {
                (self.x, self.y) = pyxis_algorithm::rotate_2d(self.x - ox, self.y - oy, &m);
                self.x += ox;
                self.y += oy;
            }
            (super::RotatePlane::Zx, _) => {
                (self.z, self.x) = pyxis_algorithm::rotate_2d(self.z - oz, self.x - ox, &m);
                self.z += oz;
                self.x += ox;
            }
            (super::RotatePlane::Yz, _) => {
                (self.y, self.z) = pyxis_algorithm::rotate_2d(self.y - oy, self.z - oz, &m);
                self.y += oy;
                self.z += oz;
            }
        };
    }
    pub fn scale(&mut self, sx: f64, sy: f64, sz: f64, ox: f64, oy: f64, oz: f64) {
        if sx == 1.0f64 && sy == 1.0f64 && sz == 1.0f64 {
            tracing::warn!("Scale parameters are all 1.");
        }
        if sx == self.x && sy == self.x && sz == self.x {
            tracing::warn!("Scale origin is equal to coordinate.");
        }
        self.x = ox + (self.x - ox) * sx;
        self.y = oy + (self.y - oy) * sy;
        self.z = oz + (self.z - oz) * sz;
    }
    pub fn space(&mut self, from: CoordSpace, to: CoordSpace) {
        (self.x, self.y, self.z) = match (from, to) {
            (CoordSpace::Cartesian, CoordSpace::Cylindrical) => {
                pyxis_algorithm::cartesian_to_cylindrical(self.x, self.y, self.z)
            }
            (CoordSpace::Cartesian, CoordSpace::Spherical) => {
                pyxis_algorithm::cartesian_to_spherical(self.x, self.y, self.z)
            }
            (CoordSpace::Cylindrical, CoordSpace::Cartesian) => {
                pyxis_algorithm::cylindrical_to_cartesian(self.x, self.y, self.z)
            }
            (CoordSpace::Cylindrical, CoordSpace::Spherical) => {
                pyxis_algorithm::cylindrical_to_spherical(self.x, self.y, self.z)
            }
            (CoordSpace::Spherical, CoordSpace::Cartesian) => {
                pyxis_algorithm::spherical_to_cartesian(self.x, self.y, self.z)
            }
            (CoordSpace::Spherical, CoordSpace::Cylindrical) => {
                pyxis_algorithm::spherical_to_cylindrical(self.x, self.y, self.z)
            }
            _ => {
                tracing::warn!("Nothing changes from <{from}> to <{to}>.");
                (self.x, self.y, self.z)
            }
        };
    }
    pub fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        if tx == 0.0f64 && ty == 0.0f64 && tz == 0.0f64 {
            tracing::warn!(
                "Translation parameters are all 0. The Coordinate is not modified after translation."
            )
        }
        self.x += tx;
        self.y += ty;
        self.z += tz;
    }
    pub fn xyz2lbh(&mut self, ellipsoid: &Ellipsoid<f64>) {
        (self.x, self.y, self.z) =
            pyxis_algorithm::xyz2lbh(self.x, self.y, self.z, ellipsoid, 1e-17, 1000);
    }
}
