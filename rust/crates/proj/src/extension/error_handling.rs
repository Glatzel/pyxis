#[macro_export]
macro_rules! check_result {
    ($self:expr) => {
        let code = $self.errno();

        match code {
            $crate::PjErrorCode::ProjSuccess => {
                clerk::debug!("Proj Process successed.");
            }
            ecode => {
                let report = $self.errno_string(&ecode);
                clerk::error!(
                    "Proj Process Failed. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                );
                miette::bail!(
                    "Proj Process Failed. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                )
            }
        }
    };
    ($self:expr,$code:expr) => {
        let code = $crate::PjErrorCode::from($code as u32);
        match code {
            $crate::PjErrorCode::ProjSuccess => {
                clerk::debug!("Proj Process successed.");
            }
            ecode => {
                let report = $self.errno_string(&ecode);
                clerk::error!(
                    "Proj Process Failed. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                );
                miette::bail!(
                    "Proj Process Failed. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                )
            }
        }
    };
}
