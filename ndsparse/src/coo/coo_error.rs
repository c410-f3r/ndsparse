use core::fmt;

/// Any error related to `Coo` operations
#[derive(Debug, PartialEq)]
pub enum CooError {
  // Coo::new
  /// Some index isn't in asceding order
  /// ```rust
  /// use ndsparse::coo::{CooArray, CooError};
  /// let coo = CooArray::new([2, 2], [([1, 1].into(), 8), ([0, 0].into(), 9)]);
  /// assert_eq!(coo, Err(ndsparse::Error::Coo(CooError::InvalidIndcsOrder)));
  /// ```
  InvalidIndcsOrder,
  /// Some index is greater than the defined dimensions
  /// ```rust
  /// use ndsparse::coo::{CooArray, CooError};
  /// let coo = CooArray::new([2, 2], [([0, 1].into(), 8), ([9, 9].into(), 9)]);
  /// assert_eq!(coo, Err(ndsparse::Error::Coo(CooError::InvalidIndcs)));
  /// ```
  InvalidIndcs,
  /// There are duplicated indices
  /// ```rust
  /// use ndsparse::coo::{CooArray, CooError};
  /// let coo = CooArray::new([2, 2], [([0, 0].into(), 8), ([0, 0].into(), 9)]);
  /// assert_eq!(coo, Err(ndsparse::Error::Coo(CooError::DuplicatedIndices)));
  DuplicatedIndices,

  // Coo::new_controlled_random_rand
  /// nnz is greater than the maximum permitted number of nnz
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::coo::{CooError, CooVec};
  /// use rand::{thread_rng, Rng};
  /// let mut rng = thread_rng();
  /// let dims = [1, 2, 3]; // Max of 6 elements (1 * 2 * 3)
  /// let coo: ndsparse::Result<CooVec<[usize; 3], u8>>;
  /// coo = CooVec::new_controlled_random_rand(dims, 10, &mut rng, |r, _| r.gen());
  /// assert_eq!(coo, Err(ndsparse::Error::Coo(CooError::NnzGreaterThanMaximumNnz)));
  /// ```
  #[cfg(feature = "with-rand")]
  NnzGreaterThanMaximumNnz,
}

impl fmt::Display for CooError {
  #[allow(clippy::use_debug)]
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "{:?}", self)
  }
}

#[cfg(feature = "std")]
impl std::error::Error for CooError {}
