macro_rules! check_result {
    ($self:expr) => {
        match $self.errno() {
            $crate::data_types::ProjErrorCode::Success => {
                clerk::debug!("Proj Process succeeded.");
            }
            ecode => {
                let message = $self.errno_string(ecode.clone());
                let err = crate::data_types::ProjError {
                    code: $self.errno(),
                    message,
                };
                clerk::error!("{}", err);
                return Err(err);
            }
        }
    };
    ($self:expr,$code:expr) => {
        let code = $crate::data_types::ProjError::from($code);
        match code {
            $crate::data_types::ProjError::Success => {
                clerk::debug!("Proj Process succeeded.");
            }
            ecode => {
                let message = $self.errno_string(ecode.clone());
                let err = crate::data_types::ProjError { code, message };
                clerk::error!("{}", err);
                Err(err)
            }
        }
    };
}
pub(crate) use check_result;
