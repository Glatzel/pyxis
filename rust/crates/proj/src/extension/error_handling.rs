macro_rules! check_result {
    ($self:expr) => {
        let code = $self.errno();
        let code_str = format!("{:?}", code);

        match code {
            $crate::data_types::ProjErrorCode::Success => {
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
    ($self:expr,$code:expr) => {
        let code = $crate::data_types::ProjError::from($code);
        let code_str = format!("{:?}", code);
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
