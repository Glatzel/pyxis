#![allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    unexpected_cfgs
)]

#[cfg(not(bindgen))]
include!("bindings.rs");
#[cfg(bindgen)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// impl serde
#[cfg(feature = "serde")]
macro_rules! impl_serde_for_proj_type {
    ($type:ty,$($field:ident: $f64:ty),+) => {
        impl serde::Serialize for $type {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                ($(self.$field),+).serialize(serializer)
            }
        }

        impl<'de> serde::Deserialize<'de> for $type {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: serde::Deserializer<'de>,
            {
                let ($($field),+) = <( $($f64,)+ )>::deserialize(deserializer)?;
                Ok(Self { $($field),+ })
            }
        }
    };
}
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_LP, lam:f64, phi:f64);
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_XY, x:f64, y:f64);
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_UV, u:f64, v:f64);

#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_LPZ, lam:f64, phi:f64, z:f64);
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_XYZ, x:f64, y:f64, z:f64);
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_UVW, u:f64, v:f64, w:f64);

#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_LPZT, lam:f64, phi:f64, z:f64, t:f64);
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_XYZT, x:f64, y:f64, z:f64, t:f64);
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_UVWT, u:f64, v:f64, w:f64, t:f64);

#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_OPK, o:f64, p:f64, k:f64);
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_ENU, e:f64, n:f64, u:f64);
#[cfg(feature = "serde")]
impl_serde_for_proj_type!(PJ_GEOD, s:f64,a1:f64, a2:f64);

use std::ptr::null_mut;

const NULL_PTR: *mut f64 = null_mut::<f64>();

pub trait IPjCoord: Clone {
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

        match (x.is_null(), y.is_null(), z.is_null(), t.is_null()) {
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
                    "Input data is not correct.x.is_null: {x},t.is_null: {y},z.is_null: {z},t.is_null: {t}"
                )
            }
        }
    }
}
