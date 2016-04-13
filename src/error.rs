use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fmt::Error as FmtError;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug)]
pub enum HALError {
  ///
  /// The function returned successfully
  ///
  SUCCESS            =   0,
  ///
  /// Gradient check error
  ///
  GRADIENT_ERROR     =   1,
  ///
  /// Unknown Error
  ///
  UNKNOWN            =   999
}

impl Display for HALError {
  fn fmt(&self, f: &mut Formatter) -> Result<(), FmtError> {
    write!(f, "{}", self.description())
  }
}

impl Error for HALError {
  fn description(&self) -> &str {
    match *self {
      HALError::SUCCESS        => "Function returned successfully",
      HALError::GRADIENT_ERROR => "Gradient check error",
      HALError::UNKNOWN        => "Unkown Error",
    }
  }
}
