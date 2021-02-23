#[cfg(feature = "with-rayon")]
/// Parallel iterator for Rayon implementation. This is mostly an internal detail.
#[derive(Debug)]
pub struct ParallelIteratorWrapper<I>(pub(crate) I);

#[cfg(feature = "with-rayon")]
/// Parallel producer for Rayon implementation. This is mostly an internal detail.
#[derive(Debug)]
pub struct ParallelProducerWrapper<I>(pub(crate) I);

#[inline]
pub(crate) fn are_in_ascending_order<'a, F, T, U>(slice: &'a [T], cb: F) -> bool
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

#[inline]
pub(crate) fn are_in_upper_bound<T>(slice: &[T], upper_bound: &T) -> bool
where
  T: PartialOrd,
{
  slice.iter().all(|x| x < upper_bound)
}

#[inline]
pub(crate) fn has_duplicates<T>(slice: &[T]) -> bool
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
pub(crate) fn max_nnz<const D: usize>(dims: &[usize; D]) -> usize {
  if dims == &cl_traits::default_array() {
    return 0;
  }
  if let Some(first) = dims.get(0).copied() {
    if D == 1 {
      return first;
    }

    let mut product: usize = 1;
    for dim in dims.iter().copied().filter(|dim| dim != &0) {
      product = product.saturating_mul(dim);
    }
    return product;
  }
  0
}

#[cfg(feature = "with-rand")]
#[inline]
pub(crate) fn valid_random_dims<R, const D: usize>(rng: &mut R, upper_bound: usize) -> [usize; D]
where
  R: rand::Rng,
{
  let dims = cl_traits::default_array();
  if D == 0 {
    return dims;
  }
  let cut_point = rng.gen_range(0, D);
  let mut array = dims;
  let iter = if let Some(r) = array.get_mut(cut_point..) {
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
pub(crate) fn windows2<T>(slice: &[T]) -> impl Iterator<Item = [&T; 2]> {
  slice.windows(2).filter_map(|value| Some([value.get(0)?, value.get(1)?]))
}
