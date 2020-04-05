use crate::{
  csl::{Csl, CslVec},
  Dims,
};
use cl_traits::{Push, Storage};

impl<DA, DATA, DS, IS, OS> quickcheck::Arbitrary for Csl<DA, DS, IS, OS>
where
  DA: Dims + Clone + Default + Send + 'static,
  DS: AsMut<[DATA]>
    + AsRef<[DATA]>
    + Clone
    + Default
    + Push<Input = DATA>
    + Send
    + Storage<Item = DATA>
    + 'static,
  IS: AsMut<[usize]> + AsRef<[usize]> + Clone + Default + Push<Input = usize> + Send + 'static,
  OS: AsMut<[usize]> + AsRef<[usize]> + Clone + Default + Push<Input = usize> + Send + 'static,
  rand::distributions::Standard: rand::distributions::Distribution<DATA>,
{
  #[inline]
  fn arbitrary<G>(g: &mut G) -> Self
  where
    G: quickcheck::Gen,
  {
    Self::new_random_with_rand(g, g.size())
  }
}

macro_rules! create_tests {
  ($mod_name:ident, $dim:expr, $previous_dim:expr) => {
    mod $mod_name {
      use crate::csl::CslVec;

      #[cfg(feature = "with-rayon")]
      #[quickcheck_macros::quickcheck]
      fn csl_outermost_rayon_iter(csl: CslVec<[usize; $dim], i32>) -> bool {
        use rayon::prelude::*;
        csl
          .outermost_rayon_iter()
          .enumerate()
          .all(|(idx, csl_ref)| csl_ref == csl.outermost_iter().nth(idx).unwrap())
      }

      #[quickcheck_macros::quickcheck]
      fn csl_outermost_sub_dim(csl: CslVec<[usize; $dim], i32>) -> bool {
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
fn csl_line(csl: CslVec<[usize; 2], i32>) -> bool {
  for (idx, csl_ref) in csl.outermost_iter().enumerate() {
    if csl_ref.sub_dim(0..csl.dims[1]) != csl.line([idx, 0]).unwrap() {
      return false;
    }
  }
  true
}

#[quickcheck_macros::quickcheck]
fn csl_value(csl: CslVec<[usize; 1], i32>) -> bool {
  for (idx, data) in csl.indcs().iter().copied().zip(csl.data()) {
    if Some(data) != csl.value([idx]) {
      return false;
    }
  }
  true
}
