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
  windows2(slice).all(|x| {
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

pub fn has_duplicates<T>(slice: &[T]) -> bool
where
  T: PartialEq,
{
  for (a_idx, a) in slice.iter().enumerate() {
    for b in slice.iter().skip(a_idx.saturating_add(1)) {
      if a == b {
        return true;
      }
    }
  }
  false
}

#[inline]
pub fn max_nnz<DA>(dims: &ArrayWrapper<DA>) -> usize
where
  DA: Dims,
{
  if dims == &ArrayWrapper::default() {
    return 0;
  }
  if let Some(first) = dims.slice().first().copied() {
    if DA::CAPACITY == 1 {
      return first;
    }

    let mut product: usize = 1;
    for dim in dims.slice().iter().copied().filter(|dim| dim != &0) {
      product = product.saturating_mul(dim);
    }
    return product;
  }
  0
}

#[cfg(feature = "with-rand")]
pub fn valid_random_dims<A, R>(rng: &mut R, upper_bound: usize) -> ArrayWrapper<A>
where
  A: Dims,
  R: rand::Rng,
{
  let dims = ArrayWrapper::default();
  if A::CAPACITY == 0 {
    return dims;
  }
  let cut_point = rng.gen_range(0, A::CAPACITY);
  let mut array: A = *dims;
  let iter = if let Some(r) = array.slice_mut().get_mut(cut_point..) {
    r.iter_mut()
  } else {
    return dims;
  };
  match upper_bound {
    0 => {}
    1 => iter.for_each(|dim| *dim = 1),
    _ => iter.for_each(|dim| *dim = rng.gen_range(1, upper_bound)),
  }
  dims
}

#[inline]
pub fn windows2<'a: 'b, 'b, T>(slice: &'a [T]) -> impl Iterator<Item = [&'b T; 2]> {
  #[allow(clippy::indexing_slicing)]
  slice.windows(2).map(|value| [&value[0], &value[1]])
}
