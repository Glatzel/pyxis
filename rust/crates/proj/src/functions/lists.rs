use envoy::PtrToString;

use crate::data_types::ProjError;

///Get a pointer to an array of ellipsoids defined in PROJ. The last entry
/// of the returned array is a NULL-entry. The array is statically
/// allocated and does not need to be freed after use.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_ellps>
pub fn list_operations() -> Result<Vec<crate::data_types::Operations>, ProjError> {
    let ptr = unsafe { proj_sys::proj_list_operations() };
    let mut out_vec = Vec::new();
    let mut offset = 0;
    assert!(!ptr.is_null());
    loop {
        let current_ptr = unsafe {
            ptr.offset(offset)
                .as_ref()
                .ok_or(ProjError::new("Invalid pointer".to_string()))?
        };
        if current_ptr.id.is_null() {
            break;
        } else {
            out_vec.push(crate::data_types::Operations::new(
                current_ptr.id.to_string()?,
                unsafe {
                    current_ptr
                        .descr
                        .offset(0)
                        .as_ref()
                        .ok_or_else(|| ProjError::new("Invalid pointer".to_string()))?
                        .to_string()?
                },
            ));
            offset += 1;
        }
    }
    Ok(out_vec)
}

///Get a pointer to an array of ellipsoids defined in PROJ. The last entry of
/// the returned array is a NULL-entry. The array is statically allocated and
/// does not need to be freed after use.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_ellps>
pub fn list_ellps() -> Result<Vec<crate::data_types::Ellps>, ProjError> {
    let ptr = unsafe { proj_sys::proj_list_ellps() };
    let mut out_vec = Vec::new();

    let mut offset = 0;
    loop {
        let current_ptr = unsafe {
            ptr.offset(offset)
                .as_ref()
                .ok_or(ProjError::new("Invalid pointer".to_string()))?
        };
        if current_ptr.id.is_null() {
            break;
        } else {
            out_vec.push(crate::data_types::Ellps::new(
                current_ptr.id.to_string().unwrap_or_default(),
                current_ptr.major.to_string().unwrap_or_default(),
                current_ptr.ell.to_string().unwrap_or_default(),
                current_ptr.name.to_string().unwrap_or_default(),
            ));
            offset += 1;
        }
    }
    Ok(out_vec)
}
///Get a pointer to an array of distance units defined in PROJ. The last entry
/// of the returned array is a NULL-entry. The array is statically allocated and
/// does not need to be freed after use.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_units>
pub fn list_units() -> Result<Vec<crate::data_types::Units>, ProjError> {
    let ptr = unsafe { proj_sys::proj_list_units() };
    let mut out_vec = Vec::new();
    let mut offset = 0;
    loop {
        let current_ptr = unsafe {
            ptr.offset(offset)
                .as_ref()
                .ok_or(ProjError::new("Invalid pointer".to_string()))?
        };
        if current_ptr.id.is_null() {
            break;
        } else {
            out_vec.push(crate::data_types::Units::new(
                current_ptr.id.to_string().unwrap_or_default(),
                current_ptr.to_meter.to_string().unwrap_or_default(),
                current_ptr.name.to_string().unwrap_or_default(),
                current_ptr.factor,
            ));
            offset += 1;
        }
    }
    Ok(out_vec)
}
///Get a pointer to an array of hard-coded prime meridians defined in PROJ.
/// Note that this list is no longer updated. The last entry of the returned
/// array is a NULL-entry. The array is statically allocated and does not need
/// to be freed after use.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_prime_meridians>
pub fn list_prime_meridians() -> Result<Vec<crate::data_types::PrimeMeridians>, ProjError> {
    let ptr = unsafe { proj_sys::proj_list_prime_meridians() };
    let mut out_vec = Vec::new();
    let mut offset = 0;
    loop {
        let current_ptr = unsafe {
            ptr.offset(offset)
                .as_ref()
                .ok_or(ProjError::new("Invalid pointer".to_string()))?
        };
        if current_ptr.id.is_null() {
            break;
        } else {
            out_vec.push(crate::data_types::PrimeMeridians::new(
                current_ptr.id.to_string().unwrap_or_default(),
                current_ptr.defn.to_string().unwrap_or_default(),
            ));
            offset += 1;
        }
    }
    Ok(out_vec)
}

// region:Test
#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_list_operations() -> mischief::Result<()> {
        let ops = list_operations()?;

        for i in &ops {
            println!("{}: {}\n", i.id(), i.descr());
        }
        assert!(!ops.is_empty());
        Ok(())
    }
    #[test]
    fn test_list_ellps() -> mischief::Result<()> {
        let ellps = list_ellps()?;

        for i in &ellps {
            println!("{i:?}");
        }
        assert!(!ellps.is_empty());
        Ok(())
    }
    #[test]
    fn test_list_units() -> mischief::Result<()> {
        let units = list_units()?;

        for i in &units {
            println!("{i:?}");
        }
        assert!(!units.is_empty());
        Ok(())
    }
    #[test]
    fn test_list_prime_meridians() -> mischief::Result<()> {
        let meridians = list_prime_meridians()?;

        for i in &meridians {
            println!("{i:?}");
        }
        assert!(!meridians.is_empty());
        Ok(())
    }
}
