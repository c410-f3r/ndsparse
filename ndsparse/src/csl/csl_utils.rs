use crate::csl::{Csl, CslMut, CslRef};
use cl_traits::{create_array, ArrayWrapper};
use core::{marker::PhantomData, ops::Range};

macro_rules! create_sub_dim {
  (
    $trait:ident
    $trait_fn:ident
    $ref:ident
    $line_fn:ident
    $sub_dim_fn:ident
    $([$mut:tt])?
) => {

#[inline]
pub fn $line_fn<'a: 'b, 'b, DATA: 'a, DS, IS, OS, const DIMS: usize>(
  csl: &'a $($mut)? Csl<DATA, DS, IS, OS, DIMS>,
  indcs: [usize; DIMS]
) -> Option<$ref<'b, DATA, 1>>
where
  DS: $trait<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  match DIMS {
    0 => None,
    _ => {
      let [offs_indcs, values] = line_offs(&csl.dims, &indcs, csl.offs.as_ref()).unwrap();
      Some($ref {
        data: &$($mut)? csl.data.$trait_fn()[values.clone()],
        dims: [*csl.dims.last().unwrap()].into(),
        indcs: &csl.indcs.as_ref()[values],
        offs: &csl.offs.as_ref()[offs_indcs],
        phantom: PhantomData
      })
    }
  }
}

#[inline]
pub fn $sub_dim_fn<'a: 'b, 'b, DATA: 'a, DS, IS, OS, const DIMS: usize, const N: usize>(
  csl: &'a $($mut)? Csl<DATA, DS, IS, OS, DIMS>,
  range: Range<usize>,
) -> $ref<'b, DATA, N>
where
  DS: $trait<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  assert!(range.start <= range.end);
  let data_ref = csl.data.$trait_fn();
  let dims_ref = &csl.dims;
  let indcs_ref = csl.indcs.as_ref();
  let offs_ref = csl.offs.as_ref();
  match N {
    0 => $ref {
      data: &$($mut)? [],
      dims: create_array(|_| 0usize).into(),
      indcs: &$($mut)? [],
      phantom: PhantomData,
      offs: &$($mut)? []
    },
    1 => {
      let [start_off_value, end_off_value] = [0, offs_ref[1] - offs_ref[0]];
      let indcs = &indcs_ref[start_off_value..end_off_value];
      let start = indcs.binary_search(&range.start).unwrap_or_else(|x| x);
      let end = indcs[start..].binary_search(&range.end).unwrap_or_else(|x| x);
      $ref {
        data: &$($mut)? data_ref[start..][..end],
        dims: create_array(|_| dims_ref[DIMS - N]).into(),
        indcs: &indcs_ref[start..][..end],
        phantom: PhantomData,
        offs: &offs_ref[0..2]
      }
    },
    _ => {
      let mut dims: ArrayWrapper<usize, N> = create_array(|idx| dims_ref[DIMS - N..][idx]).into();
      dims[0] = range.end - range.start;
      let [offs_indcs, offs_values] = outermost_offs(&dims, offs_ref, range);
      $ref {
        data: &$($mut)? data_ref[offs_values.clone()],
        dims,
        indcs: &indcs_ref[offs_values],
        phantom: PhantomData,
        offs: &offs_ref[offs_indcs],
      }
    },
  }
}

  };
}

create_sub_dim!(AsMut as_mut CslMut line_mut sub_dim_mut [mut]);
create_sub_dim!(AsRef as_ref CslRef line sub_dim);

pub fn data_idx<DATA, DS, IS, OS, const DIMS: usize>(
  csl: &Csl<DATA, DS, IS, OS, DIMS>,
  indcs: [usize; DIMS],
) -> Option<usize>
where
  DS: AsRef<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  assert!(ArrayWrapper::from(indcs) < csl.dims);
  match DIMS {
    0 => None,
    _ => {
      let innermost_idx = *indcs.last().unwrap();
      let [_, values] = line_offs(&csl.dims, &indcs, csl.offs.as_ref()).unwrap();
      let start = values.start;
      if let Ok(x) = csl.indcs.as_ref()[values].binary_search(&innermost_idx) {
        Some(start + x)
      } else {
        None
      }
    }
  }
}

#[inline]
pub fn line_offs<const DIMS: usize>(
  dims: &ArrayWrapper<usize, DIMS>,
  indcs: &[usize; DIMS],
  offs: &[usize],
) -> Option<[Range<usize>; 2]> {
  match DIMS {
    0 => None,
    1 => Some([0..2, offs[0]..offs[1]]),
    _ => {
      let diff = indcs.len() - 2;
      let mut lines = 0;
      for (idx, curr_idx) in indcs.iter().enumerate().take(diff) {
        lines += dims.iter().skip(idx + 1).rev().skip(1).product::<usize>() * curr_idx;
      }
      lines += indcs[dims.len() - 2];
      Some([lines..lines + 2, offs[lines]..offs[lines + 1]])
    }
  }
}

#[inline]
pub fn lines_num<const DIMS: usize>(dims: &ArrayWrapper<usize, DIMS>) -> usize {
  match DIMS {
    0 => 0,
    1 => 1,
    _ if dims == &ArrayWrapper::default() => 0,
    _ => dims.iter().rev().skip(1).filter(|dim| **dim != 0).product::<usize>(),
  }
}

#[inline]
pub fn max_nnz<const DIMS: usize>(dims: &ArrayWrapper<usize, DIMS>) -> usize {
  match DIMS {
    0 => 0,
    1 => dims[0],
    _ if dims == &ArrayWrapper::default() => 0,
    _ => dims.iter().filter(|dim| **dim != 0).product::<usize>(),
  }
}

#[inline]
pub(crate) fn offs_len<const DIMS: usize>(dims: &ArrayWrapper<usize, DIMS>) -> usize {
  match DIMS {
    0 => 0,
    1 => 2,
    _ if dims == &ArrayWrapper::default() => 0,
    _ => lines_num(dims) + 1,
  }
}

#[inline]
pub fn outermost_offs<const DIMS: usize>(
  dims: &ArrayWrapper<usize, DIMS>,
  offs: &[usize],
  range: Range<usize>,
) -> [Range<usize>; 2] {
  let outermost_stride = outermost_stride(&dims);
  let start_off_idx = outermost_stride * range.start;
  let end_off_idx = outermost_stride * range.end;
  [start_off_idx..end_off_idx + 1, offs[start_off_idx]..offs[end_off_idx]]
}

#[inline]
pub fn outermost_stride<const DIMS: usize>(dims: &ArrayWrapper<usize, DIMS>) -> usize {
  dims.iter().skip(1).rev().skip(1).product::<usize>()
}
