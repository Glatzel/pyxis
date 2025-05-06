use std::ptr::null_mut;

use crate::proj_sys::{PJ_COORD, PJ_XY, PJ_XYZ, PJ_XYZT};

const NULL_PTR: *mut f64 = null_mut::<f64>();

pub trait IPjCoord: Clone + Sized {
    fn x(&mut self) -> *mut f64;
    fn y(&mut self) -> *mut f64;
    fn z(&mut self) -> *mut f64;
    fn t(&mut self) -> *mut f64;
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
impl<T> From<T> for PJ_COORD
where
    T: IPjCoord,
{
    fn from(value: T) -> Self {
        let mut src = value.clone();
        let x = src.x();
        let y = src.y();
        let z = src.z();
        let t = src.t();
        let coord = match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
            //2d
            (false, false, true, true) => PJ_COORD {
                xy: PJ_XY {
                    x: unsafe { *x },
                    y: unsafe { *y },
                },
            },
            //3d
            (false, false, false, true) => PJ_COORD {
                xyz: PJ_XYZ {
                    x: unsafe { *x },
                    y: unsafe { *y },
                    z: unsafe { *z },
                },
            },
            //4d
            (false, false, false, false) => PJ_COORD {
                xyzt: PJ_XYZT {
                    x: unsafe { *x },
                    y: unsafe { *y },
                    z: unsafe { *z },
                    t: unsafe { *t },
                },
            },
            (x, y, z, t) => {
                panic!(
                    "Input data is not correct.x.is_null: {x},t.is_null:{y},z.is_null: {z},t.is_null: {t}"
                );
            }
        };
        coord
    }
}
