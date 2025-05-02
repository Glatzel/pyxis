pub trait IPjCoord: Clone + Sized {
    fn pj_x(&self) -> f64;
    fn pj_y(&self) -> f64;
    fn pj_z(&self) -> f64;
    fn pj_t(&self) -> f64;
    fn from_pj_coord(x: f64, y: f64, z: f64, t: f64) -> Self;
}
impl IPjCoord for (f64, f64) {
    fn pj_x(&self) -> f64 {
        self.0
    }

    fn pj_y(&self) -> f64 {
        self.1
    }

    fn pj_z(&self) -> f64 {
        0.0
    }

    fn pj_t(&self) -> f64 {
        f64::INFINITY
    }
    fn from_pj_coord(x: f64, y: f64, _z: f64, _t: f64) -> Self {
        (x, y)
    }
}
impl IPjCoord for [f64; 2] {
    fn pj_x(&self) -> f64 {
        self[0]
    }

    fn pj_y(&self) -> f64 {
        self[1]
    }

    fn pj_z(&self) -> f64 {
        0.0
    }

    fn pj_t(&self) -> f64 {
        f64::INFINITY
    }

    fn from_pj_coord(x: f64, y: f64, _z: f64, _t: f64) -> Self {
        [x, y]
    }
}
impl IPjCoord for (f64, f64, f64) {
    fn pj_x(&self) -> f64 {
        self.0
    }

    fn pj_y(&self) -> f64 {
        self.1
    }

    fn pj_z(&self) -> f64 {
        self.2
    }

    fn pj_t(&self) -> f64 {
        f64::INFINITY
    }

    fn from_pj_coord(x: f64, y: f64, z: f64, _t: f64) -> Self {
        (x, y, z)
    }
}
impl IPjCoord for [f64; 3] {
    fn pj_x(&self) -> f64 {
        self[0]
    }

    fn pj_y(&self) -> f64 {
        self[1]
    }

    fn pj_z(&self) -> f64 {
        self[2]
    }

    fn pj_t(&self) -> f64 {
        f64::INFINITY
    }

    fn from_pj_coord(x: f64, y: f64, z: f64, _t: f64) -> Self {
        [x, y, z]
    }
}
impl IPjCoord for (f64, f64, f64, f64) {
    fn pj_x(&self) -> f64 {
        self.0
    }

    fn pj_y(&self) -> f64 {
        self.1
    }

    fn pj_z(&self) -> f64 {
        self.2
    }

    fn pj_t(&self) -> f64 {
        self.3
    }

    fn from_pj_coord(x: f64, y: f64, z: f64, t: f64) -> Self {
        (x, y, z, t)
    }
}
impl IPjCoord for [f64; 4] {
    fn pj_x(&self) -> f64 {
        self[0]
    }

    fn pj_y(&self) -> f64 {
        self[1]
    }

    fn pj_z(&self) -> f64 {
        self[2]
    }

    fn pj_t(&self) -> f64 {
        self[3]
    }

    fn from_pj_coord(x: f64, y: f64, z: f64, t: f64) -> Self {
        [x, y, z, t]
    }
}
impl crate::Pj {
    pub fn project<T>(&self, inv: bool, coord: T) -> T
    where
        T: IPjCoord,
    {
        let direction = if inv {
            crate::PjDirection::PjInv
        } else {
            crate::PjDirection::PjFwd
        };
        let mut x = coord.pj_x();
        let mut y = coord.pj_y();
        let mut z = coord.pj_z();
        let mut t = coord.pj_t();

        self.trans_generic(
            direction, &mut x, 1, 1, &mut y, 1, 1, &mut z, 1, 1, &mut t, 1, 1,
        );
        T::from_pj_coord(x, y, z, t)
    }
    pub fn convert<T>(&self, coord: T) -> T
    where
        T: IPjCoord,
    {
        let mut x = coord.pj_x();
        let mut y = coord.pj_y();
        let mut z = coord.pj_z();
        let mut t = coord.pj_t();

        self.trans_generic(
            crate::PjDirection::PjFwd,
            &mut x,
            1,
            1,
            &mut y,
            1,
            1,
            &mut z,
            1,
            1,
            &mut t,
            1,
            1,
        );
        T::from_pj_coord(x, y, z, t)
    }
}
