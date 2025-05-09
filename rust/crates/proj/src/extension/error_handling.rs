macro_rules! check_result {
    ($self:expr) => {
        let code = $self.errno();
        let code_str = format!("{:?}", code);

        match code {
            $crate::data_types::PjError::Success => {
                clerk::debug!("Proj Process succeeded.");
            }
            ecode => {
                let report = $self.errno_string(&ecode);
                clerk::error!(
                    "Proj Process Failed. Exist code: {}<{}>. {}",
                    code_str,
                    i32::from(&ecode),
                    report
                );
                miette::bail!(
                    "Proj Process Failed. Exist code: {}<{}>. {}",
                    code_str,
                    i32::from(&ecode),
                    report
                )
            }
        }
    };
    ($self:expr,$code:expr) => {
        let code = $crate::data_types::PjError::from($code as u32);
        let code_str = format!("{:?}", code);
        match code {
            $crate::data_types::PjError::Success => {
                clerk::debug!("Proj Process succeeded.");
            }
            ecode => {
                let report = $self.errno_string(&ecode);
                clerk::error!(
                    "Proj Process Failed. Exist code: {}<{}>. {}",
                    code_str,
                    i32::from(&ecode),
                    report
                );
                miette::bail!(
                    "Proj Process Failed. Exist code: {}<{}>. {}",
                    code_str,
                    i32::from(&ecode),
                    report
                )
            }
        }
    };
}
pub(crate) use check_result;
