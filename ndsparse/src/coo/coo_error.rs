use core::fmt;

/// Any error related to `Coo` operations
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum CooError {
  /// Some index isn't in asceding order
  ///
  /// ```rust
  /// use ndsparse::coo::{CooArray, CooError};
  /// let coo = CooArray::new([2, 2], [([1, 1], 8), ([0, 0], 9)]);
  /// assert_eq!(coo, Err(ndsparse::Error::Coo(CooError::InvalidIndcsOrder)));
  /// ```
  InvalidIndcsOrder,

  /// Some index is greater than the defined dimensions
  ///
  /// ```rust
  /// use ndsparse::coo::{CooArray, CooError};
  /// let coo = CooArray::new([2, 2], [([0, 1], 8), ([9, 9], 9)]);
  /// assert_eq!(coo, Err(ndsparse::Error::Coo(CooError::InvalidIndcs)));
  /// ```
  InvalidIndcs,

  /// There are duplicated indices
  ///
  /// ```rust
  /// use ndsparse::coo::{CooArray, CooError};
  /// let coo = CooArray::new([2, 2], [([0, 0], 8), ([0, 0], 9)]);
  /// assert_eq!(coo, Err(ndsparse::Error::Coo(CooError::DuplicatedIndices)));
  DuplicatedIndices,

  /// nnz is greater than the maximum permitted number of nnz
  ///
  #[cfg_attr(all(feature = "alloc", feature = "with-rand"), doc = "```rust")]
  #[cfg_attr(not(all(feature = "alloc", feature = "with-rand")), doc = "```ignore")]
  /// use ndsparse::coo::{CooError, CooVec};
  /// use rand::{Rng, rngs::mock::StepRng};
  /// let mut rng = StepRng::new(0, 1);
  /// let dims = [1, 2, 3]; // Max of 6 elements (1 * 2 * 3)
  /// let coo: ndsparse::Result<CooVec<u8, 3>>;
  /// coo = CooVec::new_controlled_random_rand(dims, 10, &mut rng, |r, _| r.gen());
  /// assert_eq!(coo, Err(ndsparse::Error::Coo(CooError::NnzGreaterThanMaximumNnz)));
  /// ```
  #[cfg(feature = "with-rand")]
  NnzGreaterThanMaximumNnz,
}

impl fmt::Display for CooError {
  #[inline]
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let s = match *self {
      Self::InvalidIndcsOrder => "InvalidIndcsOrder",
      Self::InvalidIndcs => "InvalidIndcs",
      Self::DuplicatedIndices => "DuplicatedIndices",
      #[cfg(feature = "with-rand")]
      Self::NnzGreaterThanMaximumNnz => "NnzGreaterThanMaximumNnz",
    };
    write!(f, "{}", s)
  }
}

#[cfg(feature = "std")]
impl std::error::Error for CooError {}
