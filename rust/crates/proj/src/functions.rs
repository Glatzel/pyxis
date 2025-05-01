// region:Lists
pub fn list_operations() -> Vec<crate::PjOperations> {
    let ptr = unsafe { proj_sys::proj_list_operations() };
    let mut src_vec = Vec::new();
    let mut end_flag = false;
    let mut offset = 0;
    while !end_flag {
        match unsafe { ptr.offset(offset).as_ref() } {
            Some(src) => {
                src_vec.push(src);
            }
            None => end_flag = true,
        }
        offset += 1;
    }

    unimplemented!()
}
pub fn list_ellps() -> Vec<crate::PjEllps> {
    let ptr = unsafe { proj_sys::proj_list_ellps() };
    let mut out_vec = Vec::new();
    let mut end_flag = false;
    let mut offset = 0;

    while !end_flag {
        match unsafe { ptr.offset(offset).as_ref() } {
            Some(src) => {
                out_vec.push(crate::PjEllps::new(
                    crate::c_char_to_string(src.id),
                    crate::c_char_to_string(src.major),
                    crate::c_char_to_string(src.ell),
                    crate::c_char_to_string(src.name),
                ));
            }
            None => end_flag = true,
        }
        offset += 1;
    }
    out_vec
}
pub fn list_units() -> Vec<crate::PjUnits> {
    let ptr = unsafe { proj_sys::proj_list_units() };
    let mut out_vec = Vec::new();
    let mut end_flag = false;
    let mut offset = 0;

    while !end_flag {
        match unsafe { ptr.offset(offset).as_ref() } {
            Some(src) => {
                out_vec.push(crate::PjUnits::new(
                    crate::c_char_to_string(src.id),
                    crate::c_char_to_string(src.to_meter),
                    crate::c_char_to_string(src.name),
                    src.factor,
                ));
            }
            None => end_flag = true,
        }
        offset += 1;
    }
    out_vec
}
pub fn list_prime_meridians() -> Vec<crate::PjPrimeMeridians> {
    let ptr = unsafe { proj_sys::proj_list_prime_meridians() };
    let mut out_vec = Vec::new();
    let mut end_flag = false;
    let mut offset = 0;

    while !end_flag {
        match unsafe { ptr.offset(offset).as_ref() } {
            Some(src) => {
                out_vec.push(crate::PjPrimeMeridians::new(
                    crate::c_char_to_string(src.id),
                    crate::c_char_to_string(src.defn),
                ));
            }
            None => end_flag = true,
        }
        offset += 1;
    }
    out_vec
}
// region:Cleanup
///<https://proj.org/en/stable/development/reference/functions.html#c.proj_cleanup>
pub fn cleanup() {
    unsafe { proj_sys::proj_cleanup() };
}
// region:C API for ISO-19111 functionality
#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_list_ellps() {
        let ellps = list_ellps();

        for i in &ellps {
            println!("{:?}", i);
        }
        assert!(!ellps.is_empty());
    }
}
