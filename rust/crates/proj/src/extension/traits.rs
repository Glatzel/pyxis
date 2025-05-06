// coordinate type conversion
use std::ptr::null_mut;
const NULL_PTR: *mut f64 = null_mut::<f64>();
pub trait IPjCoord: Clone {
    fn x(&mut self) -> *mut f64;
    fn y(&mut self) -> *mut f64;
    fn z(&mut self) -> *mut f64;
    fn t(&mut self) -> *mut f64;
    fn to_coord(self) -> miette::Result<proj_sys::PJ_COORD> {
        let mut src = self.clone();
        let x = src.x();
        let y = src.y();
        let z = src.z();
        let t = src.t();

        let coord = match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => proj_sys::PJ_COORD {
                xy: proj_sys::PJ_XY {
                    x: unsafe { *x },
                    y: unsafe { *y },
                },
            },
            //3d
            (false, false, false, true) => proj_sys::PJ_COORD {
                xyz: proj_sys::PJ_XYZ {
                    x: unsafe { *x },
                    y: unsafe { *y },
                    z: unsafe { *z },
                },
            },
            //4d
            (false, false, false, false) => proj_sys::PJ_COORD {
                xyzt: proj_sys::PJ_XYZT {
                    x: unsafe { *x },
                    y: unsafe { *y },
                    z: unsafe { *z },
                    t: unsafe { *t },
                },
            },
            (x, y, z, t) => miette::bail!(format!(
                "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
            )),
        };
        Ok(coord)
    }
}
impl IPjCoord for (f64, f64) {
    fn x(&mut self) -> *mut f64 { &mut self.0 }
    fn y(&mut self) -> *mut f64 { &mut self.1 }
    fn z(&mut self) -> *mut f64 { NULL_PTR }
    fn t(&mut self) -> *mut f64 { NULL_PTR }
}
impl IPjCoord for [f64; 2] {
    fn x(&mut self) -> *mut f64 { &mut self[0] }
    fn y(&mut self) -> *mut f64 { &mut self[1] }
    fn z(&mut self) -> *mut f64 { NULL_PTR }
    fn t(&mut self) -> *mut f64 { NULL_PTR }
}
impl IPjCoord for (f64, f64, f64) {
    fn x(&mut self) -> *mut f64 { &mut self.0 }
    fn y(&mut self) -> *mut f64 { &mut self.1 }
    fn z(&mut self) -> *mut f64 { &mut self.2 }
    fn t(&mut self) -> *mut f64 { NULL_PTR }
}
impl IPjCoord for [f64; 3] {
    fn x(&mut self) -> *mut f64 { &mut self[0] }
    fn y(&mut self) -> *mut f64 { &mut self[1] }
    fn z(&mut self) -> *mut f64 { &mut self[2] }
    fn t(&mut self) -> *mut f64 { NULL_PTR }
}
impl IPjCoord for (f64, f64, f64, f64) {
    fn x(&mut self) -> *mut f64 { &mut self.0 }
    fn y(&mut self) -> *mut f64 { &mut self.1 }
    fn z(&mut self) -> *mut f64 { &mut self.2 }
    fn t(&mut self) -> *mut f64 { &mut self.3 }
}
impl IPjCoord for [f64; 4] {
    fn x(&mut self) -> *mut f64 { &mut self[0] }
    fn y(&mut self) -> *mut f64 { &mut self[1] }
    fn z(&mut self) -> *mut f64 { &mut self[2] }
    fn t(&mut self) -> *mut f64 { &mut self[3] }
}
