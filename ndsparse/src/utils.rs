#[cfg(feature = "with_rayon")]
/// Parallel iterator for Rayon implementation. This is mostly an internal detail.
#[derive(Debug)]
pub struct ParallelIteratorWrapper<I>(pub(crate) I);

#[cfg(feature = "with_rayon")]
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
