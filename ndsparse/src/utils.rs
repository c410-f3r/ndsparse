use crate::Dims;
use cl_traits::ArrayWrapper;

#[cfg(feature = "with-rayon")]
/// Parallel iterator for Rayon implementation. This is mostly an internal detail.
#[derive(Debug)]
pub struct ParallelIteratorWrapper<I>(pub(crate) I);

#[cfg(feature = "with-rayon")]
/// Parallel producer for Rayon implementation. This is mostly an internal detail.
#[derive(Debug)]
pub struct ParallelProducerWrapper<I>(pub(crate) I);

pub fn are_in_ascending_order<'a, F, T, U>(slice: &'a [T], cb: F) -> bool
where
  F: Fn(&'a T, &'a T) -> [&'a U; 2],
  T: 'a,
  U: PartialOrd + 'a,
{
  slice.windows(2).all(|x| {
    let [a, b] = cb(&x[0], &x[1]);
    a <= b
  })
}

pub fn are_in_upper_bound<T>(slice: &[T], upper_bound: &T) -> bool
where
  T: PartialOrd,
{
  slice.iter().all(|x| x < upper_bound)
}

pub fn does_not_have_duplicates<T>(slice: &[T]) -> bool
where
  T: PartialEq,
{
  for (a_idx, a) in slice.iter().enumerate() {
    for b in slice.iter().skip(a_idx + 1) {
      if a == b {
        return false;
      }
    }
  }
  true
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

#[cfg(feature = "with-rand")]
pub fn valid_random_dims<A: Dims, R: rand::Rng>(
  rng: &mut R,
  upper_bound: usize,
) -> ArrayWrapper<A> {
  let mut dims = ArrayWrapper::default();
  match A::CAPACITY {
    0 => {}
    _ => {
      let cut_point = rng.gen_range(0, A::CAPACITY);
      let iter = (&mut *dims as &mut A).slice_mut()[cut_point..].iter_mut();
      match upper_bound {
        0 => {}
        1 => iter.for_each(|dim| *dim = 1),
        _ => iter.for_each(|dim| *dim = rng.gen_range(1, upper_bound)),
      }
    }
  }
  dims
}
