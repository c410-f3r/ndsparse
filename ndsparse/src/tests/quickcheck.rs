use crate::csl::CslVec;

macro_rules! create_tests {
  ($mod_name:ident, $dim:expr) => {
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
    }
  };
}

create_tests!(_0, 0);
create_tests!(_1, 1);
create_tests!(_2, 2);

#[quickcheck_macros::quickcheck]
fn csl_value(csl: CslVec<i32, 1>) -> bool {
  for (idx, data) in csl.indcs().iter().copied().zip(csl.data()) {
    if Some(data) != csl.value([idx]) {
      return false;
    }
  }
  true
}
