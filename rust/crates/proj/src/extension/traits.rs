use std::ptr::null_mut;

pub trait IPjCoord: Clone {
    fn pj_x(&mut self) -> *mut f64;
    fn pj_y(&mut self) -> *mut f64;
    fn pj_z(&mut self) -> *mut f64;
    fn pj_t(&mut self) -> *mut f64;
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
}
