use crate::csl::{outermost_offs, CslError, CslMut, CslRef};
use core::mem;

macro_rules! impl_iter {
  ($csl_iter:ident, $data_type:ty, $split_at:ident, $ref:ident) => {
    /// Iterator of a CSL dimension.
    #[derive(Debug, PartialEq)]
    pub struct $csl_iter<'a, T, const D: usize> {
      curr_idx: usize,
      data: $data_type,
      dims: [usize; D],
      indcs: &'a [usize],
      max_idx: usize,
      offs: &'a [usize],
    }

    impl<'a, T, const D: usize> $csl_iter<'a, T, D> {
      pub(crate) fn new(
        mut dims: [usize; D],
        data: $data_type,
        indcs: &'a [usize],
        offs: &'a [usize],
      ) -> crate::Result<Self> {
        if let Some(r) = dims.first_mut() {
          let max_idx = *r;
          *r = 1;
          Ok($csl_iter { curr_idx: 0, data, dims, indcs, max_idx, offs })
        } else {
          Err(CslError::InvalidIterDim.into())
        }
      }

      #[cfg(feature = "with-rayon")]
      pub(crate) fn split_at(self, idx: usize) -> [Self; 2] {
        let cut_point = self.curr_idx + idx;
        let [_, values] = outermost_offs(&self.dims, self.offs, self.curr_idx..cut_point);
        let (data_head, data_tail) = self.data.$split_at(values.end - values.start);
        let (indcs_head, indcs_tail) = self.indcs.split_at(values.end - values.start);
        [
          $csl_iter {
            curr_idx: self.curr_idx,
            data: data_head,
            dims: self.dims,
            indcs: indcs_head,
            max_idx: cut_point,
            offs: self.offs,
          },
          $csl_iter {
            curr_idx: cut_point,
            data: data_tail,
            dims: self.dims,
            indcs: indcs_tail,
            max_idx: self.max_idx,
            offs: self.offs,
          },
        ]
      }
    }

    impl<'a, T, const D: usize> DoubleEndedIterator for $csl_iter<'a, T, D> {
      fn next_back(&mut self) -> Option<Self::Item> {
        if self.curr_idx == 0 {
          return None;
        }
        let range = self.curr_idx - 1..self.curr_idx;
        self.curr_idx -= 1;
        let [indcs, values] = outermost_offs(&self.dims, self.offs, range);
        let data = mem::take(&mut self.data);
        let (data_head, data_tail) = data.$split_at(values.end - values.start);
        let (indcs_head, indcs_tail) = self.indcs.split_at(values.end - values.start);
        self.data = data_tail;
        self.indcs = indcs_tail;
        Some($ref {
          data: data_head,
          dims: self.dims,
          indcs: indcs_head,
          offs: self.offs.get(indcs)?,
        })
      }
    }

    impl<'a, T, const D: usize> ExactSizeIterator for $csl_iter<'a, T, D> {}

    impl<'a, T, const D: usize> Iterator for $csl_iter<'a, T, D> {
      type Item = $ref<'a, T, D>;

      fn next(&mut self) -> Option<Self::Item> {
        if self.curr_idx >= self.max_idx {
          return None;
        }
        let range = self.curr_idx..self.curr_idx + 1;
        self.curr_idx += 1;
        let [indcs, values] = outermost_offs(&self.dims, self.offs, range);
        let data = mem::take(&mut self.data);
        let (data_head, data_tail) = data.$split_at(values.end - values.start);
        let (indcs_head, indcs_tail) = self.indcs.split_at(values.end - values.start);
        self.data = data_tail;
        self.indcs = indcs_tail;
        Some($ref {
          data: data_head,
          dims: self.dims,
          indcs: indcs_head,
          offs: self.offs.get(indcs)?,
        })
      }

      fn size_hint(&self) -> (usize, Option<usize>) {
        (self.max_idx, Some(self.max_idx))
      }
    }
  };
}

impl_iter!(CslLineIterMut, &'a mut [T], split_at_mut, CslMut);
impl_iter!(CslLineIterRef, &'a [T], split_at, CslRef);
