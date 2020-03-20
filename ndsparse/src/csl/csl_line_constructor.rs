use crate::{csl::Csl, utils::*, Dims};
use cl_traits::{Push, Storage};

/// Constructs valid lines in a easy and interactive manner, abstracting away the complexity
/// of the compressed sparse format.
#[derive(Debug, PartialEq)]
pub struct CslLineConstructor<'a, DA, DS, IS, PS>
where
  DA: Dims,
{
  csl: &'a mut Csl<DA, DS, IS, PS>,
  curr_dim: usize,
}

impl<'a, DA, DATA, DS, IS, PS> CslLineConstructor<'a, DA, DS, IS, PS>
where
  DA: Dims,
  DS: AsRef<[DATA]> + Push<Input = DATA> + Storage<Item = DATA>,
  IS: AsRef<[usize]> + Push<Input = usize>,
  PS: AsRef<[usize]> + Push<Input = usize>,
{
  pub(crate) fn new(csl: &'a mut Csl<DA, DS, IS, PS>) -> Self {
    let curr_dim = if let Some(idx) = csl.dims.slice().iter().copied().position(|x| x != 0) {
      idx
    } else {
      if csl.offs.as_ref().get(0).is_none() {
        csl.offs.push(0);
      }
      csl.dims.slice().len()
    };
    Self { csl, curr_dim }
  }

  /// Jumps to the next outermost dimension, i.e., from right to left.
  ///
  /// # Example
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<[usize; 3], i32>::default();
  /// csl
  ///   .constructor()
  ///   .next_outermost_dim(3)
  ///   .push_line(&[1], &[0])
  ///   .next_outermost_dim(4)
  ///   .push_line(&[2], &[1])
  ///   .push_empty_line()
  ///   .push_line(&[3, 4], &[0, 1]);
  /// assert_eq!(
  ///   csl.sub_dim(0..4),
  ///   CslRef::new([4, 3], &[1, 2, 3, 4][..], &[0, 1, 0, 1][..], &[0, 1, 2, 2, 4][..])
  /// );
  /// ```
  ///
  /// # Assertions
  ///
  /// * The next dimension must not exceed the defined number of dimensions.
  #[cfg_attr(feature = "alloc", doc = "```rust,should_panic")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::CslVec;
  /// let _ = CslVec::<[usize; 0], i32>::default().constructor().next_outermost_dim(2);
  /// ```
  pub fn next_outermost_dim(mut self, len: usize) -> Self {
    assert!(self.curr_dim != 0, "Maximum of {} dimensions", DA::CAPACITY);
    self.curr_dim -= 1;
    self.csl.dims[self.curr_dim] = len;
    self
  }

  /// This is the same as `push_line(&[], &[])`.
  ///
  /// # Example
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<[usize; 3], i32>::default();
  /// let constructor = csl.constructor();
  /// constructor.next_outermost_dim(3).push_empty_line().next_outermost_dim(2).push_empty_line();
  /// assert_eq!(csl.line([0, 0, 0]), Some(CslRef::new([3], &[][..], &[][..], &[0, 0][..])));
  /// ```
  pub fn push_empty_line(self) -> Self {
    self.csl.offs.push(*self.csl.offs.as_ref().last().unwrap());
    self
  }

  /// Pushes a new compressed line, modifying the internal structure and if applicable,
  /// increasing the current dimension length.
  ///
  /// Both `data` and `indcs` will be truncated by the length of the lesser slice.
  ///
  /// # Arguments
  ///
  /// * `data`: A slice of cloned items.
  /// * `indcs`: The respective index of each item.
  ///
  /// # Example
  ///
  #[cfg_attr(feature = "alloc", doc = "```rust")]
  #[cfg_attr(not(feature = "alloc"), doc = "```ignore")]
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<[usize; 3], i32>::default();
  /// csl.constructor().next_outermost_dim(50).push_line(&[1, 2], &[1, 40]);
  /// let line = csl.line([0, 0, 0]);
  /// assert_eq!(line, Some(CslRef::new([50], &[1, 2][..], &[1, 40][..], &[0, 2][..])));
  /// ```
  ///
  /// # Assertions
  ///
  /// Uses a subset of the assertions of the [`Csl::new`] method.
  pub fn push_line(self, data: &[DATA], indcs: &[usize]) -> Self
  where
    DATA: Clone,
  {
    let curr_dim_rev = self.csl.dims.slice().len() - self.curr_dim;
    self.csl.dims[self.curr_dim] = match curr_dim_rev {
      0 => self.csl.dims[self.curr_dim].max(*indcs.iter().max().unwrap()),
      1 => self.csl.dims[self.curr_dim].max(self.csl.offs.as_ref().len() - 1),
      _ => self.csl.dims.slice().iter().skip(1).take(curr_dim_rev).rev().skip(1).product::<usize>(),
    };
    let mut nnz = 0;
    let last_dim = *self.csl.dims.slice().last().unwrap();
    for (idx, value) in indcs.iter().copied().zip(data.iter().cloned()) {
      assert!(
        nnz < last_dim,
        "Number of Non-Zeros of a line must be less than the innermost dimension length"
      );
      self.csl.indcs.push(idx);
      self.csl.data.push(value);
      nnz += 1;
    }
    let last_off = *self.csl.offs.as_ref().last().unwrap();
    let inserted_indcs = &self.csl.indcs.as_ref()[last_off..];
    assert!(does_not_have_duplicates(inserted_indcs), "Inserted indices must be unique.");
    self.csl.offs.push(last_off + nnz);
    self
  }
}
