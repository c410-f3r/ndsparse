use crate::{
  coo::CooError,
  csl::{CslError, CslLineConstructorError},
};
use core::fmt;

/// Contains all errors related to ndsparse
#[derive(Debug, PartialEq)]
pub enum Error {
  /// CooError
  Coo(CooError),
  /// CslError
  Csl(CslError),
  /// CslLineConstructorError
  CslLineConstructor(CslLineConstructorError),
  /// This is a bug within the internal logic and shouldn't have happened
  InvalidOperation,
  /// Couldn't unwrap optional element
  NoElem,
}

#[cfg(feature = "with-rand")]
impl Error {
  pub(crate) fn opt<T>(opt: Option<T>) -> crate::Result<T> {
    if let Some(r) = opt {
      Ok(r)
    } else {
      Err(Self::NoElem)
    }
  }
}

impl fmt::Display for Error {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match *self {
      Self::Coo(ref x) => write!(f, "Coo({})", x),
      Self::Csl(ref x) => write!(f, "Csl({})", x),
      Self::CslLineConstructor(ref x) => write!(f, "CslLineConstructor({})", x),
      Self::InvalidOperation => write!(f, "InvalidOperation"),
      Self::NoElem => write!(f, "InvalidOperation"),
    }
  }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl From<CooError> for Error {
  fn from(f: CooError) -> Self {
    Self::Coo(f)
  }
}

impl From<CslError> for Error {
  fn from(f: CslError) -> Self {
    Self::Csl(f)
  }
}

impl From<CslLineConstructorError> for Error {
  fn from(f: CslLineConstructorError) -> Self {
    Self::CslLineConstructor(f)
  }
}
