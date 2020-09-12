use crate::{csl::Csl, Dims};
use cl_traits::{Push, Storage};
use core::fmt;

/// Constructs valid lines in a easy and interactive manner, abstracting away the complexity
/// of the compressed sparse format.
#[derive(Debug, PartialEq)]
pub struct CslLineConstructor<'a, DA, DS, IS, PS>
where
  DA: Dims,
{
  csl: &'a mut Csl<DA, DS, IS, PS>,
  curr_dim_idx: usize,
  last_off: usize,
}

impl<'a, DA, DATA, DS, IS, PS> CslLineConstructor<'a, DA, DS, IS, PS>
where
  DA: Dims,
  DS: AsRef<[DATA]> + Push<Input = DATA> + Storage<Item = DATA>,
  IS: AsRef<[usize]> + Push<Input = usize>,
  PS: AsRef<[usize]> + Push<Input = usize>,
{
  pub(crate) fn new(csl: &'a mut Csl<DA, DS, IS, PS>) -> crate::Result<Self> {
    if DA::CAPACITY == 0 {
      return Err(CslLineConstructorError::EmptyDimension.into());
    }
    let curr_dim_idx = if let Some(idx) = csl.dims.slice().iter().copied().position(|x| x != 0) {
      idx
    } else {
      csl.dims.slice().len()
    };
    let last_off = Self::last_off(&*csl);
    Ok(Self { csl, curr_dim_idx, last_off })
  }

  /// Jumps to the next outermost dimension, i.e., from right to left.
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// # fn main() -> ndsparse::Result<()> {
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<[usize; 3], i32>::default();
  /// csl
  ///   .constructor()?
  ///   .next_outermost_dim(3)?
  ///   .push_line([(0, 1)].iter().copied())?
  ///   .next_outermost_dim(4)?
  ///   .push_line([(1, 2)].iter().copied())?
  ///   .push_empty_line()
  ///   .push_line([(0, 3), (1,4)].iter().copied())?;
  /// assert_eq!(
  ///   csl.sub_dim(0..4),
  ///   CslRef::new([4, 3], &[1, 2, 3, 4][..], &[0, 1, 0, 1][..], &[0, 1, 2, 2, 4][..]).ok()
  /// );
  /// # Ok(()) }
  pub fn next_outermost_dim(mut self, len: usize) -> crate::Result<Self> {
    self.curr_dim_idx =
      self.curr_dim_idx.checked_sub(1).ok_or(CslLineConstructorError::DimsOverflow)?;
    *self.curr_dim() = len;
    Ok(self)
  }

  /// This is the same as `push_line([].iter(), [].iter())`.
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// # fn main() -> ndsparse::Result<()> {
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<[usize; 3], i32>::default();
  /// let constructor = csl.constructor()?.next_outermost_dim(3)?;
  /// constructor.push_empty_line().next_outermost_dim(2)?.push_empty_line();
  /// assert_eq!(csl.line([0, 0, 0]), CslRef::new([3], &[][..], &[][..], &[0, 0][..]).ok());
  /// # Ok(()) }
  pub fn push_empty_line(self) -> Self {
    self.csl.offs.push(self.last_off);
    self
  }

  /// Pushes a new compressed line, modifying the internal structure and if applicable,
  /// increases the current dimension length.
  ///
  /// The iterator will be truncated to (usize::Max - last offset value + 1) or (last dimension value)
  /// and it can lead to a situation where no values will be inserted.
  ///
  /// # Arguments
  ///
  /// * `data`:  Iterator of cloned items.
  /// * `indcs`: Iterator of the respective indices of each item.
  ///
  /// # Example
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// # fn main() -> ndsparse::Result<()> {
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<[usize; 3], i32>::default();
  /// csl.constructor()?.next_outermost_dim(50)?.push_line([(1, 1), (40, 2)].iter().copied())?;
  /// let line = csl.line([0, 0, 0]);
  /// assert_eq!(line, CslRef::new([50], &[1, 2][..], &[1, 40][..], &[0, 2][..]).ok());
  /// # Ok(()) }
  pub fn push_line<DI>(mut self, di: DI) -> crate::Result<Self>
  where
    DI: Iterator<Item = (usize, DATA)>,
  {
    let nnz_iter = 1..self.last_dim().saturating_add(1);
    let off_iter = self.last_off.saturating_add(1)..;
    let mut iter = off_iter.zip(nnz_iter.zip(di));
    let mut last_off = self.last_off;
    let mut nnz = 0;

    let mut push = |curr_last_off, curr_nnz, idx, value| {
      self.csl.indcs.push(idx);
      self.csl.data.push(value);
      nnz = curr_nnz;
      last_off = curr_last_off;
    };

    let mut last_line_idx = if let Some((curr_last_off, (curr_nnz, (idx, value)))) = iter.next() {
      push(curr_last_off, curr_nnz, idx, value);
      idx
    } else {
      return Ok(self.push_empty_line());
    };

    for (curr_last_off, (curr_nnz, (idx, value))) in iter {
      if idx <= last_line_idx {
        return Err(CslLineConstructorError::UnsortedIndices.into());
      }
      push(curr_last_off, curr_nnz, idx, value);
      last_line_idx = idx;
    }

    if nnz == 0 {
      return Ok(self.push_empty_line());
    }
    self.csl.offs.push(last_off);
    self.last_off = last_off;
    Ok(self)
  }

  // CLIPPY: self.curr_dim_idx always points to a valid reference
  #[allow(clippy::unwrap_used)]
  fn curr_dim(&mut self) -> &mut usize {
    self.csl.dims.slice_mut().get_mut(self.curr_dim_idx).unwrap()
  }

  // CLIPPY: Constructor doesn't contain empty dimensions
  #[allow(clippy::unwrap_used)]
  fn last_dim(&mut self) -> usize {
    *self.csl.dims.slice().last().unwrap()
  }

  // CLIPPY: Offsets always have at least one element
  #[allow(clippy::unwrap_used)]
  fn last_off(csl: &Csl<DA, DS, IS, PS>) -> usize {
    *csl.offs.as_ref().last().unwrap()
  }
}

/// Contains all errors related to CslLineConstructor.
#[derive(Debug, PartialEq)]
pub enum CslLineConstructorError {
  /// The maximum number of dimenstions has been reached
  DimsOverflow,
  /// All indices must be in ascending order
  UnsortedIndices,
  /// It isn't possible to construct new elements in an empty dimension
  EmptyDimension,
  /// The maximum number of lines for the currention dimension has been reached
  MaxNumOfLines,
}

impl fmt::Display for CslLineConstructorError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let s = match *self {
      Self::DimsOverflow => "DimsOverflow",
      Self::UnsortedIndices => "UnsortedIndices",
      Self::EmptyDimension => "EmptyDimension",
      Self::MaxNumOfLines => "MaxNumOfLines",
    };
    write!(f, "{}", s)
  }
}

#[cfg(feature = "std")]
impl std::error::Error for CslLineConstructorError {}
