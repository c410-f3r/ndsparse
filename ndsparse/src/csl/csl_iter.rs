use crate::csl::{outermost_offs, CslMut, CslRef};
use cl_traits::ArrayWrapper;
use core::slice::{from_raw_parts, from_raw_parts_mut};

macro_rules! impl_iter {
  ($csl_iter:ident, $data_ptr:ty, $data_type:ty, $from_raw_parts:ident, $ref:ident) => {
    /// Iterator of a CSL dimension.
    #[derive(Debug)]
    pub struct $csl_iter<'a, T, const N: usize>
    where
      T: 'a,
    {
      curr_idx: usize,
      data: $data_ptr,
      dims: ArrayWrapper<usize, N>,
      indcs: &'a [usize],
      max_idx: usize,
      offs: &'a [usize],
    }

    impl<'a, T, const N: usize> $csl_iter<'a, T, N> {
      pub(crate) fn new(
        orig_dims: &ArrayWrapper<usize, N>,
        data: $data_ptr,
        indcs: &'a [usize],
        offs: &'a [usize],
      ) -> Self {
        assert!(N > 1);
        let mut dims = *orig_dims;
        dims[0] = 1;
        let max_idx = orig_dims[0];
        $csl_iter { curr_idx: 0, data, dims, indcs, max_idx, offs }
      }

      #[cfg(feature = "with_rayon")]
      pub(crate) fn split_at(self, idx: usize) -> [Self; 2] {
        let cut_point = self.curr_idx + idx;
        [
          $csl_iter {
            curr_idx: self.curr_idx,
            data: self.data,
            dims: self.dims,
            indcs: self.indcs,
            max_idx: cut_point,
            offs: self.offs,
          },
          $csl_iter {
            curr_idx: cut_point,
            data: self.data,
            dims: self.dims,
            indcs: self.indcs,
            max_idx: self.max_idx,
            offs: self.offs,
          },
        ]
      }
    }

    impl<'a, T, const N: usize> DoubleEndedIterator for $csl_iter<'a, T, N> {
      fn next_back(&mut self) -> Option<Self::Item> {
        if self.curr_idx == 0 {
          return None;
        }
        let range = self.curr_idx - 1..self.curr_idx;
        let [indcs, values] = outermost_offs(&self.dims, self.offs, range);
        self.curr_idx -= 1;
        Some($ref {
          data: unsafe { $from_raw_parts(self.data.add(values.start), values.end - values.start) },
          dims: self.dims,
          indcs: &self.indcs[values],
          offs: &self.offs[indcs],
        })
      }
    }

    impl<'a, T, const N: usize> ExactSizeIterator for $csl_iter<'a, T, N> {}

    impl<'a, T, const N: usize> Iterator for $csl_iter<'a, T, N> {
      type Item = $ref<'a, T, N>;

      fn next(&mut self) -> Option<Self::Item> {
        if self.curr_idx >= self.max_idx {
          return None;
        }
        let range = self.curr_idx..self.curr_idx + 1;
        let [indcs, values] = outermost_offs(&self.dims, self.offs, range);
        self.curr_idx += 1;
        Some($ref {
          data: unsafe { $from_raw_parts(self.data.add(values.start), values.end - values.start) },
          dims: self.dims,
          indcs: &self.indcs[values],
          offs: &self.offs[indcs],
        })
      }

      fn size_hint(&self) -> (usize, Option<usize>) {
        (self.max_idx, Some(self.max_idx))
      }
    }

    unsafe impl<'a, T, const N: usize> Send for $csl_iter<'a, T, N> {}
    unsafe impl<'a, T, const N: usize> Sync for $csl_iter<'a, T, N> {}
  };
}

impl_iter!(CslIterMut, *mut T, &'a mut [T], from_raw_parts_mut, CslMut);
impl_iter!(CsIterRef, *const T, &'a [T], from_raw_parts, CslRef);
