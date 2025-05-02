#[macro_export]
macro_rules! check_pj_result {
    ($self:expr) => {
        let code = $self.errno();
        check_pj_result_inner!($self, code);
    };
    ($self:expr,$code:expr) => {
        let code = crate::PjErrorCode::from($code as u32);
        check_pj_result_inner!($self, code);
    };
}
#[macro_export]
macro_rules! check_pj_result_inner {
    ($self:expr, $code:expr) => {
        match $code {
            crate::PjErrorCode::ProjSuccess => {
                clerk::info!("Proj Process successed.");
            }
            ecode => {
                let report = $self.errno_string(&ecode);
                clerk::error!(
                    "Proj Process Failed. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                );
                $self.errno_reset();
                miette::bail!(
                    "Proj Process Failed. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                )
            }
        }
    };
}
#[macro_export]
macro_rules! check_context_result {
    ($self:expr) => {
        let code = $self.errno();
        check_context_result_inner!($self, code);
    };
    ($self:expr,$code:expr) => {
        let code = crate::PjErrorCode::from($code as u32);
        check_context_result_inner!($self, code);
    };
}
#[macro_export]
macro_rules! check_context_result_inner {
    ($self:expr, $code:expr) => {
        match $code {
            crate::PjErrorCode::ProjSuccess => {
                clerk::info!("Proj Process successed.");
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
