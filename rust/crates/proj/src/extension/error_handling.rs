/// Performs standardized error checking for PROJ operations.
///
/// This macro has **three forms** to handle different situations:
///
/// 1. **Check `self.errno()` for success or error**
///
/// ```ignore
/// check_result!(self);
/// ```
///
/// Expands to a `match` on `self.errno()`.
/// - If `ProjErrorCode::Success`, logs a debug message.
/// - Otherwise, it constructs a `ProjError` from `errno` and its string
///   description, logs an error, and immediately returns `Err(err)`.
///
/// 2. **Check a specific error code value**
///
/// ```ignore
/// check_result!(self, code_expr);
/// ```
///
/// Takes an expression (typically a return code) and converts it to a
/// `ProjError` with `ProjError::from`.
/// - If `Success`, logs a debug message.
/// - Otherwise, builds a `ProjError` with `errno_string` from `self` and logs
///   an error. Returns `Err(err)` (no early `return`, the caller should handle
///   the result).
///
/// 3. **Guard with a custom condition and message**
///
/// ```ignore
/// check_result!(condition_expr, message_expr);
/// ```
///
/// If `condition_expr` is true, immediately returns an `Err(ProjError)` with
/// `ProjErrorCode::Other` and a formatted message built from `message_expr`.
///
/// # Parameters
/// - `$self:ident` — usually `self` or another identifier with `errno()` and
///   `errno_string()` methods.
/// - `$code:expr` — an expression that produces a PROJ error code.
/// - `$condition:expr` — boolean condition to trigger an error return.
/// - `$message:expr` — value used to build the error message when the condition
///   fails.
///
/// # Effects
/// - Logs debug messages on success via `clerk::debug!`.
/// - Logs error messages on failure via `clerk::error!`.
/// - Constructs and returns a `crate::data_types::ProjError` on error.
///
/// # Notes
/// - In form (1), the macro uses an early `return Err(err);`.
/// - In form (2), it returns `Err(err)` as the last expression of the match
///   arm, so you can bind or propagate it manually.
/// - In form (3), it always `return`s early on error.
macro_rules! check_result {
    ($self:ident) => {
        match $self.errno() {
            $crate::data_types::ProjErrorCode::Success => {
                clerk::debug!("Proj Process succeeded.");
            }
            ecode => {
                let message = $self.errno_string(ecode.clone())?;
                let err = crate::data_types::ProjError {
                    code: $self.errno(),
                    message,
                };
                clerk::error!("{}", err);
                return Err(err);
            }
        }
    };
    ($self:ident,$code:expr) => {
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
    ($condition:expr,$message:expr) => {
        if $condition {
            return Err($crate::data_types::ProjError {
                code: $crate::data_types::ProjErrorCode::Other,
                message: format!("{}", $message),
            });
        }
    };
}
pub(crate) use check_result;
