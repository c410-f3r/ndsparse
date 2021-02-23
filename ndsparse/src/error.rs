use crate::{
  coo::CooError,
  csl::{CslError, CslLineConstructorError},
};
use core::fmt;

/// Contains all errors related to ndsparse
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum Error {
  /// CooError
  Coo(CooError),
  /// CslError
  Csl(CslError),
  /// CslLineConstructorError
  CslLineConstructor(CslLineConstructorError),
  /// The internal buffer can't store all necessary data
  InsufficientCapacity,
  /// An Unknown that probably shouldn't have happened
  UnknownError,
}

impl fmt::Display for Error {
  #[inline]
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match *self {
      Self::Coo(ref x) => write!(f, "Coo({})", x),
      Self::Csl(ref x) => write!(f, "Csl({})", x),
      Self::CslLineConstructor(ref x) => write!(f, "CslLineConstructor({})", x),
      Self::InsufficientCapacity => write!(f, "Inefficient Capacity"),
      Self::UnknownError => write!(f, "UnknownError"),
    }
  }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

impl From<CooError> for Error {
  #[inline]
  fn from(f: CooError) -> Self {
    Self::Coo(f)
  }
}

impl From<CslError> for Error {
  #[inline]
  fn from(f: CslError) -> Self {
    Self::Csl(f)
  }
}

impl From<CslLineConstructorError> for Error {
  #[inline]
  fn from(f: CslLineConstructorError) -> Self {
    Self::CslLineConstructor(f)
  }
}
