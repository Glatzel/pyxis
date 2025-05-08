use std::ptr::null_mut;
const NULL_PTR: *mut f64 = null_mut();
/// A trait representing a mutable 4D coordinate (x, y, z, t) for use with PROJ
/// FFI bindings.
///
/// This trait is intended to be implemented by custom coordinate types used in
/// conjunction with PROJ coordinate transformation functions.
///
/// # Safety Contract
///
/// - The returned pointers from `x()`, `y()`, `z()`, and `t()` must remain
///   valid and point to the respective components for the lifetime of the
///   mutable reference.
/// - The `x()` and `y()` methods **must not** return null pointers. These are
///   essential for 2D transformations.
/// - `z()` and `t()` may return null pointers for 2D/3D use cases where the
///   additional dimensions are not needed.
/// - All pointers must be aligned and safe to dereference during the lifetime
///   of the mutable borrow.
/// #[doc(hidden)]
/// # Minimal Example
/// 2D coordinate
/// ```
/// use std::ptr::null_mut;
/// #[derive(Clone)]
/// struct MyCoord {
///     x: f64,
///     y: f64,
/// }
/// impl IPjCoord for MyCoord {
///     fn x(&mut self) -> *mut f64 { &mut self.x }
///     fn y(&mut self) -> *mut f64 { &mut self.y }
///     fn z(&mut self) -> *mut f64 { null_mut() }
///     fn t(&mut self) -> *mut f64 { null_mut() }
/// }
/// ```
/// 
/// 3D coordinate
/// ```
/// use std::ptr::null_mut;
/// struct MyCoord {
///     x: f64,
///     y: f64,
///     z: f64
/// }
/// impl IPjCoord for MyCoord {
///     fn x(&mut self) -> *mut f64 { &mut self.x }
///     fn y(&mut self) -> *mut f64 { &mut self.y }
///     fn z(&mut self) -> *mut f64 { &mut self.z }
///     fn t(&mut self) -> *mut f64 { null_mut() }
/// }
/// ```
/// 4D coordinate
/// ```rust
/// #[derive(Clone)]
/// struct MyCoord {
///     x: f64,
///     y: f64,
///     z: f64,
///     t: f64,
/// }
/// impl IPjCoord for MyCoord {
///     fn x(&mut self) -> *mut f64 { &mut self.x }
///     fn y(&mut self) -> *mut f64 { &mut self.y }
///     fn z(&mut self) -> *mut f64 { &mut self.z }
///     fn t(&mut self) -> *mut f64 { &mut self.t }
/// }
/// ```
pub trait IPjCoord: Clone {
    fn x(&mut self) -> *mut f64;
    fn y(&mut self) -> *mut f64;
    fn z(&mut self) -> *mut f64;
    fn t(&mut self) -> *mut f64;
}

// Only allow use this triat in crate.
// Prevent users from modifying the to_coord fn.
pub(crate) trait ToCoord {
    fn to_coord(&self) -> miette::Result<proj_sys::PJ_COORD>;
}

impl<T> ToCoord for T
where
    T: IPjCoord,
{
    fn to_coord(&self) -> miette::Result<proj_sys::PJ_COORD> {
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
