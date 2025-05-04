use super::{CoordSpace, CryptoSpace, MigrateOption2d, RotateUnit, options};
pub struct ContextTransform {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl ContextTransform {
    pub fn crypto(&mut self, from: CryptoSpace, to: CryptoSpace) {
        (self.x, self.y) = match (from, to) {
            (CryptoSpace::BD09, CryptoSpace::GCJ02) => pyxis::crypto::crypto_exact(
                self.x,
                self.y,
                &pyxis::crypto::bd09_to_gcj02,
                &pyxis::crypto::gcj02_to_bd09,
                1e-17,
                pyxis::crypto::CryptoThresholdMode::LonLat,
                1000,
            ),
            (CryptoSpace::BD09, CryptoSpace::WGS84) => pyxis::crypto::crypto_exact(
                self.x,
                self.y,
                &pyxis::crypto::bd09_to_wgs84,
                &pyxis::crypto::wgs84_to_bd09,
                1e-17,
                pyxis::crypto::CryptoThresholdMode::LonLat,
                1000,
            ),
            (CryptoSpace::GCJ02, CryptoSpace::BD09) => pyxis::crypto::gcj02_to_bd09(self.x, self.y),
            (CryptoSpace::GCJ02, CryptoSpace::WGS84) => pyxis::crypto::crypto_exact(
                self.x,
                self.y,
                &pyxis::crypto::gcj02_to_wgs84,
                &pyxis::crypto::wgs84_to_gcj02,
                1e-17,
                pyxis::crypto::CryptoThresholdMode::LonLat,
                1000,
            ),
            (CryptoSpace::WGS84, CryptoSpace::BD09) => pyxis::crypto::wgs84_to_bd09(self.x, self.y),
            (CryptoSpace::WGS84, CryptoSpace::GCJ02) => {
                pyxis::crypto::wgs84_to_gcj02(self.x, self.y)
            }
            _ => {
                clerk::warn!("Nothing changes from <{from}> to <{to}>.");
                (self.x, self.y)
            }
        };
    }

    pub fn datum_compense(&mut self, hb: f64, r: f64, x0: f64, y0: f64) {
        (self.x, self.y) = pyxis::datum_compense(
            self.x,
            self.y,
            &pyxis::DatumCompenseParms::new(hb, r, x0, y0),
        );
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

        let rotate_matrix = pyxis::rotate_matrix_2d(rotate);
        (self.x, self.y) = match (given, another) {
            (MigrateOption2d::Absolute, MigrateOption2d::Origin) => {
                pyxis::migrate::rel_2d(another_x, another_y, self.x, self.y, &rotate_matrix)
            }
            (MigrateOption2d::Origin, MigrateOption2d::Absolute) => {
                pyxis::migrate::rel_2d(self.x, self.y, another_x, another_y, &rotate_matrix)
            }
            (MigrateOption2d::Absolute, MigrateOption2d::Relative) => {
                pyxis::migrate::origin_2d(self.x, self.y, another_x, another_y, &rotate_matrix)
            }
            (MigrateOption2d::Relative, MigrateOption2d::Absolute) => {
                pyxis::migrate::origin_2d(another_x, another_y, self.x, self.y, &rotate_matrix)
            }
            (MigrateOption2d::Relative, MigrateOption2d::Origin) => {
                pyxis::migrate::abs_2d(another_x, another_y, self.x, self.y, &rotate_matrix)
            }
            (MigrateOption2d::Origin, MigrateOption2d::Relative) => {
                pyxis::migrate::abs_2d(self.x, self.y, another_x, another_y, &rotate_matrix)
            }

            (given, another) => {
                clerk::warn!("Given and anther is the same. Given: {given}, Another: {another}.");
                (self.x, self.y)
            }
        }
    }
    pub fn normalize(&mut self) {
        if self.x == 0.0f64 && self.y == 0.0f64 && self.z == 0.0f64 {
            clerk::warn!("Length of coordinate vector is 0.")
        } else {
            let length = (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt();
            self.x /= length;
            self.y /= length;
            self.z /= length;
        }
    }
    pub fn proj(&mut self, from: &str, to: &str) -> miette::Result<()> {
        let ctx = crate::proj_util::init_proj_builder()?;
        let pj = ctx.create_crs_to_crs(from, to, &proj::PjArea::default())?;
        let pj = ctx.normalize_for_visualization(&pj).unwrap();
        (self.x, self.y) = pj.convert(&(self.x, self.y))?;
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
            options::RotateUnit::Degrees => pyxis::rotate_matrix_2d(r.to_radians()),
            _ => pyxis::rotate_matrix_2d(r),
        };

        match (plane, unit) {
            (_, RotateUnit::Degrees) if r % 360.0 == 0.0 => clerk::warn!(
                "Rotate angle mod 360 equals 0. The Coordinate is not modified after rotate."
            ),
            (_, RotateUnit::Radians) if r % 360.0 == 0.0 => clerk::warn!(
                "Rotate radians mod PI equals 0. The Coordinate is not modified after rotate."
            ),
            (super::RotatePlane::Xy, _) if ox == self.x && oy == self.y => clerk::warn!(
                "Rotate origin equals to coordinate. The Coordinate is not modified after rotate."
            ),
            (super::RotatePlane::Zx, _) if ox == self.x && oz == self.z => clerk::warn!(
                "Rotate origin equals to coordinate. The Coordinate is not modified after rotate."
            ),
            (super::RotatePlane::Yz, _) if oy == self.y && oz == self.z => clerk::warn!(
                "Rotate origin equals to coordinate. The Coordinate is not modified after rotate."
            ),
            (super::RotatePlane::Xy, _) => {
                (self.x, self.y) = pyxis::rotate_2d(self.x - ox, self.y - oy, &m);
                self.x += ox;
                self.y += oy;
            }
            (super::RotatePlane::Zx, _) => {
                (self.z, self.x) = pyxis::rotate_2d(self.z - oz, self.x - ox, &m);
                self.z += oz;
                self.x += ox;
            }
            (super::RotatePlane::Yz, _) => {
                (self.y, self.z) = pyxis::rotate_2d(self.y - oy, self.z - oz, &m);
                self.y += oy;
                self.z += oz;
            }
        };
    }
    pub fn scale(&mut self, sx: f64, sy: f64, sz: f64, ox: f64, oy: f64, oz: f64) {
        if sx == 1.0f64 && sy == 1.0f64 && sz == 1.0f64 {
            clerk::warn!("Scale parameters are all 1.");
        }
        if sx == self.x && sy == self.x && sz == self.x {
            clerk::warn!("Scale origin is equal to coordinate.");
        }
        self.x = ox + (self.x - ox) * sx;
        self.y = oy + (self.y - oy) * sy;
        self.z = oz + (self.z - oz) * sz;
    }
    pub fn space(&mut self, from: CoordSpace, to: CoordSpace) {
        (self.x, self.y, self.z) = match (from, to) {
            (CoordSpace::Cartesian, CoordSpace::Cylindrical) => {
                pyxis::cartesian_to_cylindrical(self.x, self.y, self.z)
            }
            (CoordSpace::Cartesian, CoordSpace::Spherical) => {
                pyxis::cartesian_to_spherical(self.x, self.y, self.z)
            }
            (CoordSpace::Cylindrical, CoordSpace::Cartesian) => {
                pyxis::cylindrical_to_cartesian(self.x, self.y, self.z)
            }
            (CoordSpace::Cylindrical, CoordSpace::Spherical) => {
                pyxis::cylindrical_to_spherical(self.x, self.y, self.z)
            }
            (CoordSpace::Spherical, CoordSpace::Cartesian) => {
                pyxis::spherical_to_cartesian(self.x, self.y, self.z)
            }
            (CoordSpace::Spherical, CoordSpace::Cylindrical) => {
                pyxis::spherical_to_cylindrical(self.x, self.y, self.z)
            }
            _ => {
                clerk::warn!("Nothing changes from <{from}> to <{to}>.");
                (self.x, self.y, self.z)
            }
        };
    }
    pub fn translate(&mut self, tx: f64, ty: f64, tz: f64) {
        if tx == 0.0f64 && ty == 0.0f64 && tz == 0.0f64 {
            clerk::warn!(
                "Translation parameters are all 0. The Coordinate is not modified after translation."
            )
        }
        self.x += tx;
        self.y += ty;
        self.z += tz;
    }
}
