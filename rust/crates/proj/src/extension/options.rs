use std::ffi::c_char;

use super::string_to_c_char;
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
    options: Vec<String>,
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
    pub fn as_ptr(&mut self) -> *const *const c_char {
        let c_options: Vec<*const c_char> = self
            .options
            .iter()
            .map(|s| string_to_c_char(s).unwrap())
            .collect();
        c_options.as_ptr()
    }
}
