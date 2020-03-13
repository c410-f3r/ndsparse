use crate::{csl::Csl, utils::*};
use cl_traits::{Push, Storage};

/// Constructs valid lines in a easy and interactive manner, abstracting away the complexity
/// of the compressed sparse format.
#[derive(Debug, PartialEq)]
pub struct CslLineConstructor<'a, DS, IS, PS, const DIMS: usize> {
  csl: &'a mut Csl<DS, IS, PS, DIMS>,
  curr_dim: usize,
}

impl<'a, DATA, DS, IS, PS, const DIMS: usize> CslLineConstructor<'a, DS, IS, PS, DIMS>
where
  DS: AsRef<[DATA]> + Push<Input = DATA> + Storage<Item = DATA>,
  IS: AsRef<[usize]> + Push<Input = usize>,
  PS: AsRef<[usize]> + Push<Input = usize>,
{
  pub(crate) fn new(csl: &'a mut Csl<DS, IS, PS, DIMS>) -> Self {
    let curr_dim = if let Some(idx) = csl.dims.iter().copied().position(|x| x != 0) {
      idx
    } else {
      if csl.offs.as_ref().get(0).is_none() {
        csl.offs.push(0);
      }
      csl.dims.len()
    };
    Self { csl, curr_dim }
  }

  /// Jumps to the next outermost dimension, i.e., from right to left.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<_, 3>::default();
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
  /// ```rust,should_panic
  /// use ndsparse::csl::CslVec;
  /// let _ = CslVec::<i32, 0>::default().constructor().next_outermost_dim(2);
  /// ```
  pub fn next_outermost_dim(mut self, len: usize) -> Self {
    assert!(self.curr_dim != 0, "Maximum of {} dimensions", DIMS);
    self.curr_dim -= 1;
    self.csl.dims[self.curr_dim] = len;
    self
  }

  /// This is the same as `push_line(&[], &[])`.
  ///
  /// # Example
  ///
  /// ```rust
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<i32, 3>::default();
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
  /// ```rust
  /// use ndsparse::csl::{CslRef, CslVec};
  /// let mut csl = CslVec::<i32, 3>::default();
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
    let curr_dim_rev = self.csl.dims.len() - self.curr_dim;
    self.csl.dims[self.curr_dim] = match curr_dim_rev {
      0 => self.csl.dims[self.curr_dim].max(*indcs.iter().max().unwrap()),
      1 => self.csl.dims[self.curr_dim].max(self.csl.offs.as_ref().len() - 1),
      _ => self.csl.dims.iter().skip(1).take(curr_dim_rev).rev().skip(1).product::<usize>(),
    };
    let mut nnz = 0;
    let last_dim = *self.csl.dims.last().unwrap();
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
