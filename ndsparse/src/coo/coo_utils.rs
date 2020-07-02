use crate::{utils::windows2, Dims};
use cl_traits::ArrayWrapper;

macro_rules! create_value {
  ($get:ident $fn_name:ident $([$mut:tt])?) => {
    pub fn $fn_name<DA, DATA>(
      indcs: ArrayWrapper<DA>,
      data: &$($mut)? [(ArrayWrapper<DA>, DATA)],
    ) -> Option<&$($mut)? DATA>
    where
      DA: Dims
    {
      if let Ok(idx) = data.binary_search_by(|value| value.0.cmp(&indcs)) {
        Some(&$($mut)? data.$get(idx)?.1)
      } else {
        None
      }
    }
  }
}

create_value!(get value);
create_value!(get_mut value_mut [mut]);

pub fn does_not_have_duplicates_sorted<F, T>(slice: &[T], mut cb: F) -> bool
where
  F: FnMut(&T, &T) -> bool,
{
  windows2(slice).all(|[a, b]| cb(a, b))
}
