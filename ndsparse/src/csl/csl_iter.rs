use crate::{
  csl::{outermost_offs, CslMut, CslRef},
  Dims,
};
use cl_traits::ArrayWrapper;
use core::slice::{from_raw_parts, from_raw_parts_mut};

macro_rules! impl_iter {
  ($csl_iter:ident, $data_ptr:ty, $data_type:ty, $from_raw_parts:ident, $ref:ident) => {
    /// Iterator of a CSL dimension.
    #[derive(Debug)]
    pub struct $csl_iter<'a, DA, T>
    where
      DA: Dims,
      T: 'a,
    {
      curr_idx: usize,
      data: $data_ptr,
      dims: ArrayWrapper<DA>,
      indcs: &'a [usize],
      max_idx: usize,
      offs: &'a [usize],
    }

    impl<'a, DA, T> $csl_iter<'a, DA, T>
    where
      DA: Dims,
    {
      pub(crate) fn new(
        orig_dims: &ArrayWrapper<DA>,
        data: $data_ptr,
        indcs: &'a [usize],
        offs: &'a [usize],
      ) -> Self {
        assert!(DA::CAPACITY > 1);
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

    impl<'a, DA, T> DoubleEndedIterator for $csl_iter<'a, DA, T>
    where
      DA: Dims,
    {
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

    impl<'a, DA, T> ExactSizeIterator for $csl_iter<'a, DA, T> where DA: Dims {}

    impl<'a, DA, T> Iterator for $csl_iter<'a, DA, T>
    where
      DA: Dims,
    {
      type Item = $ref<'a, DA, T>;

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

    unsafe impl<'a, DA, T> Send for $csl_iter<'a, DA, T> where DA: Dims {}
    unsafe impl<'a, DA, T> Sync for $csl_iter<'a, DA, T> where DA: Dims {}
  };
}

impl_iter!(CslIterMut, *mut T, &'a mut [T], from_raw_parts_mut, CslMut);
impl_iter!(CsIterRef, *const T, &'a [T], from_raw_parts, CslRef);
