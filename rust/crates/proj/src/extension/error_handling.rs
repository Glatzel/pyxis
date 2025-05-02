impl crate::Pj {
    pub(crate) fn check_exit_code(&self, name: &str, code: i32) -> miette::Result<()> {
        let code = crate::PjErrorCode::from(code as u32);
        match code {
            crate::PjErrorCode::ProjSuccess => {
                clerk::info!("Proj Process successed: {name}.");
            }
            ecode => {
                let report = self.errno_string(&ecode);
                clerk::info!(
                    "Proj Process Failed: {name}. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                );
                self.errno_reset();
                miette::bail!(
                    "Proj Process Failed: {name}. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                )
            }
        }
        Ok(())
    }
    pub(crate) fn check_result(&self, name: &str) -> miette::Result<()> {
        let code = self.errno();
        match code {
            crate::PjErrorCode::ProjSuccess => {
                clerk::info!("Proj Process successed: {name}.");
            }
            ecode => {
                let report = self.errno_string(&ecode);
                clerk::info!(
                    "Proj Process Failed: {name}. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                );
                self.errno_reset();
                miette::bail!(
                    "Proj Process Failed: {name}. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                )
            }
        }
        Ok(())
    }
}
impl crate::PjContext {
    pub(crate) fn check_exit_code(&self, name: &str, code: i32) -> miette::Result<()> {
        let code = crate::PjErrorCode::from(code as u32);
        match code {
            crate::PjErrorCode::ProjSuccess => {
                clerk::info!("Proj Process successed: {name}.");
            }
            ecode => {
                let report = self.errno_string(&ecode);
                clerk::info!(
                    "Proj Process Failed: {name}. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                );
                miette::bail!(
                    "Proj Process Failed: {name}. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                )
            }
        }
        Ok(())
    }
    pub(crate) fn check_result(&self, name: &str) -> miette::Result<()> {
        let code = self.errno();
        match code {
            crate::PjErrorCode::ProjSuccess => {
                clerk::info!("Proj Process successed: {name}.");
            }
            ecode => {
                let report = self.errno_string(&ecode);
                clerk::info!(
                    "Proj Process Failed: {name}. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                );
                miette::bail!(
                    "Proj Process Failed: {name}. Exist code: {}. {}",
                    i32::from(&ecode),
                    report
                )
            }
        }
        Ok(())
    }
}
