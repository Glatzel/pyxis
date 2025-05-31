use std::ffi::{CStr, CString};
pub(crate) trait CstrToString {
    fn to_string(&self) -> Option<String>;
}
impl CstrToString for *const i8 {
    fn to_string(&self) -> Option<String> {
        if self.is_null() {
            return None;
        }
        Some(
            unsafe { CStr::from_ptr(*self) }
                .to_string_lossy()
                .to_string(),
        )
    }
}

pub(crate) trait CstrListToVecString {
    fn to_vec_string(&self) -> Option<Vec<String>>;
}
impl CstrListToVecString for *mut *mut i8 {
    fn to_vec_string(&self) -> Option<Vec<String>> {
        if self.is_null() {
            return None;
        }
        let mut vec_str = Vec::new();
        let mut offset = 0;

        loop {
            let current_ptr = unsafe { self.offset(offset).as_ref().unwrap() };
            if current_ptr.is_null() {
                break;
            }
            vec_str.push(current_ptr.cast_const().to_string().unwrap());
            offset += 1;
        }
        Some(vec_str)
    }
}

pub(crate) trait ToCString {
    fn to_cstring(&self) -> miette::Result<CString>;
}
impl ToCString for &str {
    fn to_cstring(&self) -> miette::Result<CString> { CString::new(*self).into_diagnostic() }
}
impl ToCString for String {
    fn to_cstring(&self) -> miette::Result<CString> {
        CString::new(self.as_str()).into_diagnostic()
    }
}

pub(crate) fn pj_obj_list_to_vec(
    ctx: &Context,
    result: *const proj_sys::PJ_OBJ_LIST,
) -> miette::Result<Vec<Proj>> {
    if result.is_null() {
        miette::bail!("Error");
    }
    let count = unsafe { proj_sys::proj_list_get_count(result) };
    let mut proj_list = Vec::with_capacity(count as usize);
    for i in 0..count {
        proj_list.push(ctx.list_get(result, i)?);
    }
    unsafe {
        proj_sys::proj_list_destroy(result.cast_mut());
    }
    Ok(proj_list)
}

macro_rules! readonly_struct {
    ($name:ident, $($struct_doc:expr)+, $({$field:ident: $type:ty $(, $field_doc:expr)?}),*) => {
        $(#[doc=$struct_doc])+
        #[cfg_attr(feature = "serde", derive(serde::Serialize))]
        #[derive(Debug)]
        pub struct $name {
            $( $field: $type ),*
        }

        impl $name {
            // Constructor function to initialize the struct
            #[allow(dead_code)]
            pub fn new($($field: $type),*) -> Self {
                $name {
                    $( $field ),*
                }
            }

            // Getter methods for each field
            $(
                $(#[doc=$field_doc])?
                pub fn $field(&self) -> &$type {
                    &self.$field
                }
            )*
        }
    }
}
use miette::IntoDiagnostic;
pub(crate) use readonly_struct;

use crate::{Context, Proj};
impl crate::Proj<'_> {
    /// Panic if a `Proj` object is not CRS.
    pub fn assert_crs(&self) -> miette::Result<&Self> {
        if !self.is_crs() {
            miette::bail!("Proj object is not CRS.");
        }
        Ok(self)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_assert_crs() -> miette::Result<()> {
        let ctx = crate::new_test_ctx()?;
        //is crs
        {
            let pj = ctx.create("EPSG:4326")?;
            assert!(pj.assert_crs().is_ok());
        }
        //not crs
        {
            let pj = ctx.create("+proj=utm +zone=32 +datum=WGS84")?;
            assert!(pj.assert_crs().is_err());
        }
        Ok(())
    }
}
