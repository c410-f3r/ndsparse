use crate::csl::CslVec;

macro_rules! create_tests {
  ($mod_name:ident, $dim:expr, $previous_dim:expr) => {
    mod $mod_name {
      use crate::csl::CslVec;

      #[cfg(feature = "with_rayon")]
      #[quickcheck_macros::quickcheck]
      fn csl_outermost_rayon_iter(csl: CslVec<i32, $dim>) -> bool {
        use rayon::prelude::*;
        csl
          .outermost_rayon_iter()
          .enumerate()
          .all(|(idx, csl_ref)| csl_ref == csl.outermost_iter().nth(idx).unwrap())
      }

      #[quickcheck_macros::quickcheck]
      fn csl_outermost_sub_dim(csl: CslVec<i32, $dim>) -> bool {
        for (idx, csl_ref) in csl.outermost_iter().enumerate() {
          if csl_ref != csl.sub_dim(idx..idx + 1) {
            return false;
          }
        }
        true
      }
    }
  };
}

create_tests!(_2, 2, 1);
create_tests!(_3, 3, 2);

#[quickcheck_macros::quickcheck]
fn csl_line(csl: CslVec<i32, 2>) -> bool {
  for (idx, csl_ref) in csl.outermost_iter().enumerate() {
    if csl_ref.sub_dim(0..csl.dims[1]) != csl.line([idx, 0]).unwrap() {
      return false;
    }
  }
  true
}

#[quickcheck_macros::quickcheck]
fn csl_value(csl: CslVec<i32, 1>) -> bool {
  for (idx, data) in csl.indcs().iter().copied().zip(csl.data()) {
    if Some(data) != csl.value([idx]) {
      return false;
    }
  }
  true
}
