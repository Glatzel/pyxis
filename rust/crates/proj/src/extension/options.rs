use std::ffi::CString;
use std::ptr::null;

pub(crate) const OPTION_YES: &str = "YES";
pub(crate) const OPTION_NO: &str = "NO";
pub(crate) trait ToProjOptionString {
    fn to_option_string(&self) -> String;
}
impl ToProjOptionString for bool {
    fn to_option_string(&self) -> String {
        if *self {
            "YES".to_string()
        } else {
            "NO".to_string()
        }
    }
}

// Macro to implement ToProjOptionString for types that can use to_string()
macro_rules! impl_to_option_string {
    ($t:ty) => {
        impl ToProjOptionString for $t {
            fn to_option_string(&self) -> String { self.to_string() }
        }
    };
}

// Use macro for simple types
impl_to_option_string!(f64);
impl_to_option_string!(usize);
impl_to_option_string!(&str);

pub(crate) struct ProjOptions {
    pub(crate) options: Vec<CString>,
}
impl ProjOptions {
    pub fn new(capacity: usize) -> ProjOptions {
        Self {
            options: Vec::with_capacity(capacity),
        }
    }

    pub fn _push<T: ToProjOptionString>(&mut self, opt: T, name: &str) -> &mut Self {
        self.options.push(
            CString::new(format!("{name}={}", opt.to_option_string()))
                .expect("Error creating CString"),
        );

        self
    }
    pub fn push_optional<T: ToProjOptionString>(
        &mut self,
        opt: Option<T>,
        name: &str,
        default_value: &str,
    ) -> &mut Self {
        match opt {
            Some(opt) => {
                self.options.push(
                    CString::new(format!("{name}={}", opt.to_option_string()))
                        .expect("Error creating CString"),
                );
            }
            None => {
                self.options.push(
                    CString::new(format!("{name}={default_value}"))
                        .expect("Error creating CString"),
                );
            }
        }
        self
    }
    pub fn push_optional_pass<T: ToProjOptionString>(
        &mut self,
        opt: Option<T>,
        name: &str,
    ) -> &mut Self {
        if let Some(opt) = opt {
            self.options.push(
                CString::new(format!("{name}={}", opt.to_option_string()))
                    .expect("Error creating CString"),
            );
        }
        self
    }

    pub fn vec_ptr(&self) -> Vec<*const i8> {
        let mut ptrs = self.options.iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        ptrs.push(null());
        ptrs
    }
}
