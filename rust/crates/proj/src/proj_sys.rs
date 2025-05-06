#![allow(
    dead_code,
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case
)]
#[cfg(any(not(feature = "bindgen"), update = "true"))]
include!("./proj_sys/bindings.rs");
#[cfg(all(feature = "bindgen", update = "false"))]
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
