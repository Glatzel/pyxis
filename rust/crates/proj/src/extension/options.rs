use std::ffi::CString;

use miette::IntoDiagnostic;

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
    pub(crate) options: Vec<CString>,
}
impl PjOptions {
    pub fn new(capacity: usize) -> PjOptions {
        Self {
            options: Vec::with_capacity(capacity),
        }
    }

    pub fn _push<T: ToPjOptionString>(&mut self, opt: T, name: &str) -> miette::Result<&mut Self> {
        self.options
            .push(CString::new(format!("{name}={}", opt.to_option_string())).into_diagnostic()?);

        Ok(self)
    }
    pub fn push_optional<T: ToPjOptionString>(
        &mut self,
        opt: Option<T>,
        name: &str,
        default_value: &str,
    ) -> miette::Result<&mut Self> {
        match opt {
            Some(opt) => {
                self.options.push(
                    CString::new(format!("{name}={}", opt.to_option_string())).into_diagnostic()?,
                );
            }
            None => {
                self.options
                    .push(CString::new(format!("{name}={default_value}")).into_diagnostic()?);
            }
        }
        Ok(self)
    }
    pub fn push_optional_pass<T: ToPjOptionString>(
        &mut self,
        opt: Option<T>,
        name: &str,
    ) -> miette::Result<&mut Self> {
        match opt {
            Some(opt) => {
                self.options.push(
                    CString::new(format!("{name}={}", opt.to_option_string())).into_diagnostic()?,
                );
            }
            None => (),
        }
        Ok(self)
    }
}
