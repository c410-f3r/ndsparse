use cl_traits::ArrayWrapper;

macro_rules! create_value {
  ($fn_name:ident $([$mut:tt])?) => {
    pub fn $fn_name<DATA, const DIMS: usize>(
      indcs: ArrayWrapper<usize, DIMS>,
      data: &$($mut)? [(ArrayWrapper<usize, DIMS>, DATA)],
    ) -> Option<&$($mut)? DATA> {
      if let Ok(idx) = data.binary_search_by(|value| value.0.cmp(&indcs)) {
        Some(&$($mut)? data[idx].1)
      } else {
        None
      }
    }
  }
}

create_value!(value);
create_value!(value_mut [mut]);

pub fn does_not_have_duplicates_sorted<F, T>(slice: &[T], mut cb: F) -> bool
where
  F: FnMut(&T, &T) -> bool,
{
  slice.windows(2).all(|slice| cb(&slice[0], &slice[1]))
}
