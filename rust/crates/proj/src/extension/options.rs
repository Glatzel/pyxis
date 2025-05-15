use std::ffi::{CString, c_char};

pub(crate) const PJ_OPTION_YES: &str = "YES";
pub(crate) const PJ_OPTION_NO: &str = "NO";
pub(crate) trait ToPjOptionString {
    fn to_option_string(&self) -> String;
}
impl ToPjOptionString for bool {
    fn to_option_string(&self) -> String {
        if *self {
            "YES".to_string()
        } else {
            "NO".to_string()
        }
    }
}
impl ToPjOptionString for f64 {
    fn to_option_string(&self) -> String { self.to_string() }
}
impl ToPjOptionString for &str {
    fn to_option_string(&self) -> String { self.to_string() }
}
impl ToPjOptionString for usize {
    fn to_option_string(&self) -> String { self.to_string() }
}
pub(crate) struct PjOptions {
    pub(crate) options: Vec<String>,
}
impl PjOptions {
    pub fn new(capacity: usize) -> PjOptions {
        Self {
            options: Vec::with_capacity(capacity),
        }
    }

    pub fn _push<T: ToPjOptionString>(&mut self, opt: T, name: &str) -> &mut Self {
        self.options
            .push(format!("{name}={}", opt.to_option_string()));

        self
    }
    pub fn push_optional<T: ToPjOptionString>(
        &mut self,
        opt: Option<T>,
        name: &str,
        default_value: &str,
    ) -> &mut Self {
        match opt {
            Some(opt) => {
                self.options
                    .push(format!("{name}={}", opt.to_option_string()));
            }
            None => {
                self.options.push(format!("{name}={default_value}"));
            }
        }
        self
    }
    pub fn push_optional_pass<T: ToPjOptionString>(
        &mut self,
        opt: Option<T>,
        name: &str,
    ) -> &mut Self {
        match opt {
            Some(opt) => {
                self.options
                    .push(format!("{name}={}", opt.to_option_string()));
            }
            None => (),
        }
        self
    }
    pub fn to_cvec_ptr(self) -> (*const *const i8, Vec<CString>) {
        clerk::debug!("Options: {:?}", self.options);
        let cstrings: Vec<CString> = self
            .options
            .into_iter()
            .map(|s| CString::new(s).expect("CString::new failed"))
            .collect();
        // Convert CStrings to *const c_char
        let ptrs: Vec<*const c_char> = cstrings.iter().map(|cs| cs.as_ptr()).collect();
        (ptrs.as_ptr(), cstrings)
    }
}
