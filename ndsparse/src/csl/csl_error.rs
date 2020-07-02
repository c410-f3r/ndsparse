use core::fmt;

/// Any error related to Csl operations
#[derive(Debug, PartialEq)]
pub enum CslError {
  /// Data or indices length is greater than the product of all dimensions length
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{CslError, CslVec};
  /// let csl = CslVec::new([3], vec![8, 9, 9, 9, 9], vec![0, 5, 5, 5, 5], vec![0, 2]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::DataIndcsLengthGreaterThanDimsLength)));
  /// ```
  DataIndcsLengthGreaterThanDimsLength,

  /// The data length is different than the indices length
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{ CslError, CslVec};
  /// let csl = CslVec::new([10], vec![8, 9], vec![0], vec![0, 2]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::DiffDataIndcsLength)));
  /// ```
  DiffDataIndcsLength,

  /// Duplicated indices in a line
  /// ```rust
  /// use ndsparse::csl::{CslArray, CslError};
  /// let csl = CslArray::new([10], [8, 9], [0, 0], [0, 2]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::DuplicatedIndices)));
  /// ```
  DuplicatedIndices,

  /// A index is greater or equal to the innermost dimension length
  ///
  /// ```rust
  /// use ndsparse::csl::{CslArray, CslError};
  /// let csl = CslArray::new([10], [8, 9], [0, 10], [0, 2]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::IndcsGreaterThanEqualDimLength)));
  /// ```
  IndcsGreaterThanEqualDimLength,

  /// Some innermost dimension length is equal to zero
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{CslError, CslVec};
  /// let csl: ndsparse::Result<CslVec<[usize; 5], i32>>;
  /// csl = CslVec::new([1, 2, 3, 0, 5], vec![], vec![], vec![]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::InnermostDimsZero)));
  /// ```
  InnermostDimsZero,

  /// Line iterator must deal with non-empty dimensions
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{CslVec, CslError};
  /// let csl = CslVec::<[usize; 0], i32>::default();
  /// assert_eq!(csl.outermost_line_iter(), Err(ndsparse::Error::Csl(CslError::InvalidIterDim)));
  /// ```
  InvalidIterDim,

  /// Offsets length is different than the dimensions product
  /// (without the innermost dimension) plus one.
  /// This rule doesn't not apply to an empty dimension.
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{CslError, CslVec};
  /// let csl = CslVec::new([10], vec![8, 9], vec![0, 5], vec![0, 2, 4]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::InvalidOffsetsLength)));
  /// ```
  InvalidOffsetsLength,

  /// Offsets aren't in ascending order
  ///
  /// ```rust
  /// use ndsparse::csl::{CslArray, CslError};
  /// let csl = CslArray::new([10], [8, 9], [0, 5], [2, 0]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::InvalidOffsetsOrder)));
  /// ```
  InvalidOffsetsOrder,

  /// Last offset is not equal to the nnz
  ///
  /// ```rust
  /// use ndsparse::csl::{CslArray, CslError};
  /// let csl = CslArray::new([10], [8, 9], [0, 5], [0, 4]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::LastOffsetDifferentNnz)));
  /// ```
  LastOffsetDifferentNnz,

  /// nnz is greater than the maximum permitted number of nnz
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// use rand::{thread_rng, Rng};
  /// let mut rng = thread_rng();
  /// let dims = [1, 2, 3]; // Max of 6 elements (1 * 2 * 3)
  /// let csl: ndsparse::Result<CslVec<[usize; 3], i32>>;
  /// csl = CslVec::new_controlled_random_rand(dims, 7, &mut rng, |r, _| r.gen());
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(ndsparse::csl::CslError::NnzGreaterThanMaximumNnz)));
  /// ```
  #[cfg(feature = "with-rand")]
  NnzGreaterThanMaximumNnz,

  /// It isn't possible to have more lines than usize::MAX - 2
  ///
  /// ```rust
  /// use ndsparse::csl::{CslArray, CslError};
  /// let csl = CslArray::new([18446744073709551295, 255, 3026418949592973312], [0], [0], [0, 1]);
  /// assert_eq!(csl, Err(ndsparse::Error::Csl(CslError::OffsLengthOverflow)));
  /// ```
  OffsLengthOverflow,
}

impl fmt::Display for CslError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let s = match self {
      Self::DataIndcsLengthGreaterThanDimsLength => "DataIndcsLengthGreaterThanDimsLength",
      Self::DiffDataIndcsLength => "DiffDataIndcsLength",
      Self::DuplicatedIndices => "DuplicatedIndices",
      Self::IndcsGreaterThanEqualDimLength => "IndcsGreaterThanEqualDimLength",
      Self::InnermostDimsZero => "InnermostDimsZero",
      Self::InvalidIterDim => "InvalidIterDim",
      Self::InvalidOffsetsLength => "InvalidOffsetsLength",
      Self::InvalidOffsetsOrder => "InvalidOffsetsOrder",
      Self::LastOffsetDifferentNnz => "LastOffsetDifferentNnz",
      Self::NnzGreaterThanMaximumNnz => "NnzGreaterThanMaximumNnz",
      Self::OffsLengthOverflow => "OffsLengthOverflowb",
    };
    write!(f, "{}", s)
  }
}

#[cfg(feature = "std")]
impl std::error::Error for CslError {}
