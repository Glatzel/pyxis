use envoy::CStrToString;
fn _list_operations() -> Vec<crate::data_types::_Operations> { todo!() }
///Get a pointer to an array of ellipsoids defined in PROJ. The last entry of
/// the returned array is a NULL-entry. The array is statically allocated and
/// does not need to be freed after use.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_ellps>
pub fn list_ellps() -> Vec<crate::data_types::Ellps> {
    let ptr = unsafe { proj_sys::proj_list_ellps() };
    let mut out_vec = Vec::new();

    let mut offset = 0;
    loop {
        let current_ptr = unsafe { ptr.offset(offset).as_ref().unwrap() };
        if current_ptr.id.is_null() {
            break;
        } else {
            let src = unsafe { ptr.offset(offset).as_ref().unwrap() };
            out_vec.push(crate::data_types::Ellps::new(
                src.id.to_string().unwrap_or_default(),
                src.major.to_string().unwrap_or_default(),
                src.ell.to_string().unwrap_or_default(),
                src.name.to_string().unwrap_or_default(),
            ));
            offset += 1;
        }
    }
    out_vec
}
///Get a pointer to an array of distance units defined in PROJ. The last entry
/// of the returned array is a NULL-entry. The array is statically allocated and
/// does not need to be freed after use.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_units>
pub fn list_units() -> Vec<crate::data_types::Units> {
    let ptr = unsafe { proj_sys::proj_list_units() };
    let mut out_vec = Vec::new();
    let mut offset = 0;
    loop {
        let current_ptr = unsafe { ptr.offset(offset).as_ref().unwrap() };
        if current_ptr.id.is_null() {
            break;
        } else {
            let src = unsafe { ptr.offset(offset).as_ref().unwrap() };
            out_vec.push(crate::data_types::Units::new(
                src.id.to_string().unwrap_or_default(),
                src.to_meter.to_string().unwrap_or_default(),
                src.name.to_string().unwrap_or_default(),
                src.factor,
            ));
            offset += 1;
        }
    }
    out_vec
}
///Get a pointer to an array of hard-coded prime meridians defined in PROJ.
/// Note that this list is no longer updated. The last entry of the returned
/// array is a NULL-entry. The array is statically allocated and does not need
/// to be freed after use.
///
/// # References
/// * <https://proj.org/en/stable/development/reference/functions.html#c.proj_list_prime_meridians>
pub fn list_prime_meridians() -> Vec<crate::data_types::PrimeMeridians> {
    let ptr = unsafe { proj_sys::proj_list_prime_meridians() };
    let mut out_vec = Vec::new();
    let mut offset = 0;
    loop {
        let current_ptr = unsafe { ptr.offset(offset).as_ref().unwrap() };
        if current_ptr.id.is_null() {
            break;
        } else {
            let src = unsafe { ptr.offset(offset).as_ref().unwrap() };
            out_vec.push(crate::data_types::PrimeMeridians::new(
                src.id.to_string().unwrap_or_default(),
                src.defn.to_string().unwrap_or_default(),
            ));
            offset += 1;
        }
    }
    out_vec
}

// region:Test
#[cfg(test)]
mod test {
    use super::*;
    // region:Lists
    #[test]
    fn test_list_ellps() {
        let ellps = list_ellps();

        for i in &ellps {
            println!("{:?}", i);
        }
        assert!(!ellps.is_empty());
    }
    #[test]
    fn test_list_units() {
        let units = list_units();

        for i in &units {
            println!("{:?}", i);
        }
        assert!(!units.is_empty());
    }
    #[test]
    fn test_list_prime_meridians() {
        let meridians = list_prime_meridians();

        for i in &meridians {
            println!("{:?}", i);
        }
        assert!(!meridians.is_empty());
    }
}
