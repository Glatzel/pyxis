#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

#[cfg(not(feature = "bindgen"))]
include!("bindings.rs");
#[cfg(feature = "bindgen")]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

// impl serde
#[cfg(feature = "serde")]
use serde;

#[cfg(feature = "serde")]
impl serde::Serialize for PJ_LP {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        (self.lam, self.phi).serialize(serializer)
    }
}
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PJ_LP {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let (lam, phi) = <(f64, f64)>::deserialize(deserializer)?;
        Ok(Self { lam, phi })
    }
}
