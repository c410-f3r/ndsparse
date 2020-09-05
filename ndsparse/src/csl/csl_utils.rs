use crate::{
  csl::{Csl, CslError, CslMut, CslRef},
  Dims,
};
use cl_traits::{create_array_rslt, ArrayWrapper};
use core::ops::Range;

macro_rules! create_sub_dim {
  (
    $trait:ident
    $trait_fn:ident
    $ref:ident
    $get:ident
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
  let last_dim = if let Some(r) = csl.dims.slice().last() {
    *r
  }
  else {
    return None;
  };
  let [offs_indcs, offs_values] = line_offs(&csl.dims, &indcs, csl.offs.as_ref())?;
  Some($ref {
    data: csl.data.$trait_fn().$get(offs_values.clone())?,
    dims: [last_dim].into(),
    indcs: &csl.indcs.as_ref().get(offs_values)?,
    offs: &csl.offs.as_ref().get(offs_indcs)?,
  })
}

#[inline]
pub fn $sub_dim_fn<'a: 'b, 'b, DATA: 'a, DS, FROMDA, IS, OS, TODA>(
  csl: &'a $($mut)? Csl<FROMDA, DS, IS, OS>,
  range: Range<usize>,
) -> Option<$ref<'b, TODA, DATA>>
where
  DS: $trait<[DATA]>,
  FROMDA: Dims,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
  TODA: Dims,
{
  if range.start > range.end || TODA::CAPACITY > FROMDA::CAPACITY {
    return None;
  }
  let data_ref = csl.data.$trait_fn();
  let dims_ref = &csl.dims;
  let indcs_ref = csl.indcs.as_ref();
  let offs_ref = csl.offs.as_ref();
  match TODA::CAPACITY {
    0 => None,
    1 => {
      let [start_off_value, end_off_value] = [0, offs_ref.get(1)? - offs_ref.first()?];
      let indcs = indcs_ref.get(start_off_value..end_off_value)?;
      let start = indcs.binary_search(&range.start).unwrap_or_else(|x| x);
      let end = indcs.get(start..)?.binary_search(&range.end).unwrap_or_else(|x| x);
      let dims_ref_idx = FROMDA::CAPACITY - TODA::CAPACITY;
      let dims_array: TODA = create_array_rslt::<TODA, _, _>(|_| {
        dims_ref.slice().get(dims_ref_idx).copied().ok_or(())
      }).ok()?.into();
      Some($ref {
        data: data_ref.$get(start..)?.$get(..end)?,
        dims: dims_array.into(),
        indcs: &indcs_ref.get(start..)?.get(..end)?,
        offs: &offs_ref.get(0..2)?
      })
    },
    _ => {
      let dims_ref_lower_bound = FROMDA::CAPACITY - TODA::CAPACITY;
      let mut dims: ArrayWrapper<TODA>;
      dims = create_array_rslt::<TODA, _, _>(|idx| {
        let fun = || Some(*dims_ref.slice().get(dims_ref_lower_bound..)?.get(idx)?);
        fun().ok_or(())
      }).ok()?.into();
      *dims.slice_mut().first_mut()? = range.end - range.start;
      let [offs_indcs, offs_values] = outermost_offs(&dims, offs_ref, range);
      Some($ref {
        data: data_ref.$get(offs_values.clone())?,
        dims,
        indcs: &indcs_ref.get(offs_values)?,
        offs: &offs_ref.get(offs_indcs)?,
      })
    },
  }
}

  };
}

create_sub_dim!(AsMut as_mut CslMut get_mut line_mut sub_dim_mut [mut]);
create_sub_dim!(AsRef as_ref CslRef get line sub_dim);

// Max offset length is usize::MAX - 1
#[inline]
pub(crate) fn correct_offs_len<DA>(dims: &ArrayWrapper<DA>) -> crate::Result<usize>
where
  DA: Dims,
{
  match DA::CAPACITY {
    0 => Ok(1),
    1 => Ok(2),
    _ if dims == &ArrayWrapper::default() => Ok(1),
    _ => {
      let mut offs_len: usize = 1;
      for dim in dims.slice().iter().copied().rev().skip(1).filter(|dim| dim != &0) {
        offs_len = offs_len.saturating_mul(dim);
      }
      offs_len.checked_add(1).ok_or_else(|| CslError::OffsLengthOverflow.into())
    }
  }
}

pub fn data_idx<DA, DATA, DS, IS, OS>(csl: &Csl<DA, DS, IS, OS>, indcs: DA) -> Option<usize>
where
  DA: Dims,
  DS: AsRef<[DATA]>,
  IS: AsRef<[usize]>,
  OS: AsRef<[usize]>,
{
  let innermost_idx = indcs.slice().last()?;
  let [_, offs_values] = line_offs(&csl.dims, &indcs, csl.offs.as_ref())?;
  let start = offs_values.start;
  if let Ok(x) = csl.indcs.as_ref().get(offs_values)?.binary_search(&innermost_idx) {
    Some(start + x)
  } else {
    None
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
    1 => Some({
      let off_end = offs.get(1)?.saturating_sub(*offs.get(0)?);
      [0..2, 0..off_end]
    }),
    _ => {
      let diff = indcs.slice().len().saturating_sub(2);
      let mut lines: usize = 0;
      for (idx, curr_idx) in indcs.slice().iter().copied().enumerate().take(diff) {
        let product = dims.slice().iter().skip(idx + 1).rev().skip(1).product::<usize>();
        lines = lines.saturating_add(product.saturating_mul(curr_idx));
      }
      lines = lines.saturating_add(*indcs.slice().get(dims.slice().len() - 2)?);
      if lines > usize::MAX.saturating_sub(2) {
        return None;
      }
      let first = *offs.first()?;
      let off_start = offs.get(lines)?.saturating_sub(first);
      let off_end = offs.get(lines + 1)?.saturating_sub(first);
      Some([lines..lines.saturating_add(2), off_start..off_end])
    }
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
  let start_off_idx = outermost_stride.saturating_mul(range.start);
  let end_off_idx = outermost_stride.saturating_mul(range.end);
  let off_start = *offs.get(start_off_idx).unwrap_or(&0);
  let off_end = *offs.get(end_off_idx).unwrap_or(&0);
  [start_off_idx..end_off_idx.saturating_add(1), off_start..off_end]
}

#[inline]
pub fn outermost_stride<DA>(dims: &ArrayWrapper<DA>) -> usize
where
  DA: Dims,
{
  dims.slice().iter().skip(1).rev().skip(1).product::<usize>()
}
