use std::ptr::null_mut;

pub trait IPjCoord: Clone {
    fn pj_x(&mut self) -> *mut f64;
    fn pj_y(&mut self) -> *mut f64;
    fn pj_z(&mut self) -> *mut f64;
    fn pj_t(&mut self) -> *mut f64;
    fn to_array4(&self) -> [f64; 4];
    fn from_array4(x: f64, y: f64, z: f64, t: f64) -> Self;
}
impl IPjCoord for (f64, f64) {
    fn pj_x(&mut self) -> *mut f64 {
        &mut self.0
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self.1
    }

    fn pj_z(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }

    fn pj_t(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }

    fn to_array4(&self) -> [f64; 4] {
        [self.0, self.1, f64::NAN, f64::NAN]
    }
    fn from_array4(x: f64, y: f64, _z: f64, _t: f64) -> Self {
        (x, y)
    }
}
impl IPjCoord for [f64; 2] {
    fn pj_x(&mut self) -> *mut f64 {
        &mut self[0]
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self[1]
    }

    fn pj_z(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }

    fn pj_t(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }

    fn to_array4(&self) -> [f64; 4] {
        [self[0], self[1], f64::NAN, f64::NAN]
    }
    fn from_array4(x: f64, y: f64, _z: f64, _t: f64) -> Self {
        [x, y]
    }
}
impl IPjCoord for (f64, f64, f64) {
    fn pj_x(&mut self) -> *mut f64 {
        &mut self.0
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self.1
    }

    fn pj_z(&mut self) -> *mut f64 {
        &mut self.2
    }

    fn pj_t(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }

    fn to_array4(&self) -> [f64; 4] {
        [self.0, self.1, self.2, f64::NAN]
    }
    fn from_array4(x: f64, y: f64, z: f64, _t: f64) -> Self {
        (x, y, z)
    }
}
impl IPjCoord for [f64; 3] {
    fn pj_x(&mut self) -> *mut f64 {
        &mut self[0]
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self[1]
    }

    fn pj_z(&mut self) -> *mut f64 {
        &mut self[2]
    }

    fn pj_t(&mut self) -> *mut f64 {
        null_mut::<f64>()
    }
    fn to_array4(&self) -> [f64; 4] {
        [self[0], self[1], self[2], f64::NAN]
    }
    fn from_array4(x: f64, y: f64, z: f64, _t: f64) -> Self {
        [x, y, z]
    }
}
impl IPjCoord for (f64, f64, f64, f64) {
    fn pj_x(&mut self) -> *mut f64 {
        &mut self.0
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self.1
    }

    fn pj_z(&mut self) -> *mut f64 {
        &mut self.2
    }

    fn pj_t(&mut self) -> *mut f64 {
        &mut self.3
    }
    fn to_array4(&self) -> [f64; 4] {
        [self.0, self.1, self.2, self.3]
    }
    fn from_array4(x: f64, y: f64, z: f64, t: f64) -> Self {
        (x, y, z, t)
    }
}
impl IPjCoord for [f64; 4] {
    fn pj_x(&mut self) -> *mut f64 {
        &mut self[0]
    }

    fn pj_y(&mut self) -> *mut f64 {
        &mut self[1]
    }

    fn pj_z(&mut self) -> *mut f64 {
        &mut self[2]
    }

    fn pj_t(&mut self) -> *mut f64 {
        &mut self[3]
    }
    fn to_array4(&self) -> [f64; 4] {
        self.clone()
    }
    fn from_array4(x: f64, y: f64, z: f64, t: f64) -> Self {
        [x, y, z, t]
    }
}
