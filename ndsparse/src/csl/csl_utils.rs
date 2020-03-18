use crate::{
  csl::{Csl, CslMut, CslRef},
  Dims,
};
use cl_traits::{create_array, ArrayWrapper};
use core::ops::Range;

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
pub fn $line_fn<'a: 'b, 'b, DA, DATA: 'a, DS, IS, OS>(
  csl: &'a $($mut)? Csl<DA, DS, IS, OS>,
  indcs: DA
) -> Option<$ref<'b, [usize; 1], DATA>>
where
  DA: Dims,
  DS: $trait<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  match DA::CAPACITY {
    0 => None,
    _ => {
      let [offs_indcs, values] = line_offs(&csl.dims, &indcs, csl.offs.as_ref()).unwrap();
      Some($ref {
        data: &$($mut)? csl.data.$trait_fn()[values.clone()],
        dims: [*csl.dims.slice().last().unwrap()].into(),
        indcs: &csl.indcs.as_ref()[values],
        offs: &csl.offs.as_ref()[offs_indcs],
      })
    }
  }
}

#[inline]
pub fn $sub_dim_fn<'a: 'b, 'b, DATA: 'a, DS, FROMDA, IS, OS, TODA>(
  csl: &'a $($mut)? Csl<FROMDA, DS, IS, OS>,
  range: Range<usize>,
) -> $ref<'b, TODA, DATA>
where
  DS: $trait<[DATA]>,
  FROMDA: Dims,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
  TODA: Dims,
{
  assert!(range.start <= range.end);
  let data_ref = csl.data.$trait_fn();
  let dims_ref = &csl.dims;
  let indcs_ref = csl.indcs.as_ref();
  let offs_ref = csl.offs.as_ref();
  match TODA::CAPACITY {
    0 => $ref {
      data: &$($mut)? [],
      dims: create_array::<TODA, _>(|_| 0usize).into(),
      indcs: &$($mut)? [],
      offs: &$($mut)? []
    },
    1 => {
      let [start_off_value, end_off_value] = [0, offs_ref[1] - offs_ref[0]];
      let indcs = &indcs_ref[start_off_value..end_off_value];
      let start = indcs.binary_search(&range.start).unwrap_or_else(|x| x);
      let end = indcs[start..].binary_search(&range.end).unwrap_or_else(|x| x);
      let dims_ref_idx = FROMDA::CAPACITY - TODA::CAPACITY;
      let dims_array: TODA = create_array::<TODA, _>(|_| dims_ref[dims_ref_idx]).into();
      $ref {
        data: &$($mut)? data_ref[start..][..end],
        dims: dims_array.into(),
        indcs: &indcs_ref[start..][..end],
        offs: &offs_ref[0..2]
      }
    },
    _ => {
      let dims_ref_lower_bound = FROMDA::CAPACITY - TODA::CAPACITY;
      let mut dims: ArrayWrapper<TODA>;
      dims = create_array::<TODA, _>(|idx| dims_ref[dims_ref_lower_bound..][idx]).into();
      dims[0] = range.end - range.start;
      let [offs_indcs, offs_values] = outermost_offs(&dims, offs_ref, range);
      $ref {
        data: &$($mut)? data_ref[offs_values.clone()],
        dims,
        indcs: &indcs_ref[offs_values],
        offs: &offs_ref[offs_indcs],
      }
    },
  }
}

  };
}

create_sub_dim!(AsMut as_mut CslMut line_mut sub_dim_mut [mut]);
create_sub_dim!(AsRef as_ref CslRef line sub_dim);

pub fn data_idx<DA, DATA, DS, IS, OS>(csl: &Csl<DA, DS, IS, OS>, indcs: DA) -> Option<usize>
where
  DA: Dims,
  DS: AsRef<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  assert!(indcs.slice() < csl.dims.slice());
  match DA::CAPACITY {
    0 => None,
    _ => {
      let innermost_idx = *indcs.slice().last().unwrap();
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
pub fn line_offs<DA>(
  dims: &ArrayWrapper<DA>,
  indcs: &DA,
  offs: &[usize],
) -> Option<[Range<usize>; 2]>
where
  DA: Dims,
{
  match DA::CAPACITY {
    0 => None,
    1 => Some([0..2, offs[0]..offs[1]]),
    _ => {
      let diff = indcs.slice().len() - 2;
      let mut lines = 0;
      for (idx, curr_idx) in indcs.slice().iter().enumerate().take(diff) {
        lines += dims.slice().iter().skip(idx + 1).rev().skip(1).product::<usize>() * curr_idx;
      }
      lines += indcs.slice()[dims.slice().len() - 2];
      Some([lines..lines + 2, offs[lines]..offs[lines + 1]])
    }
  }
}

#[inline]
pub fn lines_num<DA>(dims: &ArrayWrapper<DA>) -> usize
where
  DA: Dims,
{
  match DA::CAPACITY {
    0 => 0,
    1 => 1,
    _ if dims == &ArrayWrapper::default() => 0,
    _ => dims.slice().iter().rev().skip(1).filter(|dim| **dim != 0).product::<usize>(),
  }
}

#[inline]
pub fn max_nnz<DA>(dims: &ArrayWrapper<DA>) -> usize
where
  DA: Dims,
{
  match DA::CAPACITY {
    0 => 0,
    1 => dims[0],
    _ if dims == &ArrayWrapper::default() => 0,
    _ => dims.slice().iter().filter(|dim| **dim != 0).product::<usize>(),
  }
}

#[inline]
pub(crate) fn offs_len<DA>(dims: &ArrayWrapper<DA>) -> usize
where
  DA: Dims,
{
  match DA::CAPACITY {
    0 => 0,
    1 => 2,
    _ if dims == &ArrayWrapper::default() => 0,
    _ => lines_num(dims) + 1,
  }
}

#[inline]
pub fn outermost_offs<DA>(
  dims: &ArrayWrapper<DA>,
  offs: &[usize],
  range: Range<usize>,
) -> [Range<usize>; 2]
where
  DA: Dims,
{
  let outermost_stride = outermost_stride(&dims);
  let start_off_idx = outermost_stride * range.start;
  let end_off_idx = outermost_stride * range.end;
  [start_off_idx..end_off_idx + 1, offs[start_off_idx]..offs[end_off_idx]]
}

#[inline]
pub fn outermost_stride<DA>(dims: &ArrayWrapper<DA>) -> usize
where
  DA: Dims,
{
  dims.slice().iter().skip(1).rev().skip(1).product::<usize>()
}
